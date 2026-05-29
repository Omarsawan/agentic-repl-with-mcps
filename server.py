"""FastAPI web server that exposes MCPHost as a chat UI."""
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"


def make_app(host) -> FastAPI:
    """Create and return the FastAPI app wired to the given MCPHost instance."""

    app = FastAPI()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/reset")
    async def reset() -> dict:
        host._history.clear()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        input_queue: asyncio.Queue[str] = asyncio.Queue()
        # Strong refs required — asyncio only weakly references tasks and will GC (garbage collector)
        # a suspended task before it resumes from input_queue.get().
        active_tasks: set[asyncio.Task] = set()

        async def emit(event: dict) -> None:
            await ws.send_json(event)

        async def prompt_fn(prompt: str, *, is_announcement: bool = False) -> str:
            if is_announcement:
                await emit({"type": "input_announcement", "content": prompt})
                return ""
            await emit({"type": "input_request", "prompt": prompt})
            return await input_queue.get()

        def spawn_turn(content: str) -> None:
            async def _run() -> None:
                try:
                    await host.handle_input(content, emit=emit)
                except Exception as exc:
                    await emit({"type": "error", "message": str(exc)})

            task = asyncio.create_task(_run())
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        host.set_prompt_fn(prompt_fn)

        try:
            while True:
                data = await ws.receive_json()
                if data.get("type") == "input_response":
                    # "input_response" is sent by the browser when the user submits
                    # a value via the inline argument-collection form. Route it back
                    # to prompt_fn, which is suspended waiting on input_queue.get().
                    await input_queue.put(data.get("value", ""))
                elif content := data.get("content", "").strip():
                    # process turn normally
                    spawn_turn(content)
        except WebSocketDisconnect:
            pass
        finally:
            host.set_prompt_fn(None)

    return app
