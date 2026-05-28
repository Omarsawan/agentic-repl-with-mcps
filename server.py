"""FastAPI web server that exposes MCPHost as a chat UI."""
from contextlib import asynccontextmanager
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
        try:
            while True:
                data = await ws.receive_json()
                content = data.get("content", "").strip()
                if not content:
                    continue
                try:
                    await host._turn(content, emit=ws.send_json)
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            pass

    return app
