# File: /workspace/video_ocr_engine/backend/main.py
import uvicorn
import asyncio
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.engine import OCREngine
import shutil
import os

app = FastAPI(title="OMEGA Video OCR Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global queue for SSE events
clients = []

async def broadcast_event(event_type: str, data: dict):
    message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    for client in clients:
        await client.put(message)

@app.post("/api/v1/process")
async def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(run_engine, file_location)
    return {"message": "Job started", "file": file.filename}

async def run_engine(video_path: str):
    async def progress_callback(msg: str, percent: float):
        await broadcast_event("progress", {"msg": msg, "percent": percent})
        
    try:
        engine = OCREngine(video_path, progress_callback)
        output = await engine.run_pipeline()
        
        # Send final results
        await broadcast_event("complete", output.dict())
        
        # Cleanup
        engine.cleanup()
        if os.path.exists(video_path):
            os.remove(video_path)
            
    except Exception as e:
        await broadcast_event("error", {"msg": str(e)})

@app.get("/api/v1/stream")
async def sse_stream():
    queue = asyncio.Queue()
    clients.append(queue)
    
    async def event_generator():
        try:
            while True:
                msg = await queue.get()
                yield msg
        except asyncio.CancelledError:
            clients.remove(queue)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
