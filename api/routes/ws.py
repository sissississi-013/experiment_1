import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.routes.runs import get_active_bus

router = APIRouter(tags=["websocket"])

@router.websocket("/api/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str):
    bus = get_active_bus(run_id)
    if not bus:
        await websocket.close(code=4004, reason="Run not active")
        return
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_event(event):
        data = json.loads(event.model_dump_json())
        data["type"] = type(event).__name__
        loop.call_soon_threadsafe(queue.put_nowait, data)

    bus.subscribe_all(on_event)
    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=60.0)
                await websocket.send_json(data)
                if data.get("type") == "ModuleCompleted" and data.get("module") == "reporter":
                    await websocket.send_json({"type": "RunCompleted", "run_id": run_id})
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
