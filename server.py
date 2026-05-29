"""FastAPI web server that exposes MCPHost as a chat UI."""
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from enums import EventType

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

        async def notify_fn(text: str) -> None:
            await emit({"type": EventType.AGENT_NOTIFICATION, "content": text})

        async def prompt_fn(prompt: str) -> str:
            await emit({"type": EventType.INPUT_PROMPT, "prompt": prompt})
            return await input_queue.get()

        def spawn_turn(content: str) -> None:
            async def _run() -> None:
                try:
                    await host.handle_input(content, emit=emit)
                except Exception as exc:
                    await emit({"type": EventType.ERROR, "message": str(exc)})

            task = asyncio.create_task(_run())
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        host.set_interaction_fns(notify_fn, prompt_fn)

        try:
            while True:
                data = await ws.receive_json()
                if data.get("type") == EventType.INPUT_RESPONSE:
                    # Sent by the browser when the user submits the inline argument form.
                    # Routes the value back to prompt_fn, which is suspended on input_queue.get().
                    await input_queue.put(data.get("value", ""))
                elif content := data.get("content", "").strip():
                    # process turn normally
                    spawn_turn(content)
        except WebSocketDisconnect:
            pass
        finally:
            host.set_interaction_fns(None, None)

    return app
