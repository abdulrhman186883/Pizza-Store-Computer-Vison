import cv2
import numpy as np
import torch
import time
import threading
import os
from typing import List, Union
from fastapi import FastAPI, Response, Body
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
MODEL_PATH = "yolo12m-v2.pt" 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)

class VideoState:
    def __init__(self):
        self.source: Union[str, int] = "3.mp4"
        self.roi_polygon = None
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None 
        # Violation logic variables 
        self.total_violations = 0
        self.hand_timers = {}
        self.already_counted = {}
        self.last_violation_time = 0

    def update_source(self, new_source):
        with self.lock:
            # 1. Reset Violation Counters
            self.total_violations = 0
            self.hand_timers = {}
            self.already_counted = {}
            self.last_violation_time = 0
            
            # 2. Update Video Source
            self.source = new_source
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

state = VideoState()

class Point(BaseModel):
    x: float  
    y: float

# --- LOGIC HELPERS ---
def is_inside_roi(box, polygon):
    center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
    return cv2.pointPolygonTest(polygon, center, False) >= 0

def is_colliding(h_box, target_boxes):
    for t in target_boxes:
        if not (h_box[2] < t[0]-15 or h_box[0] > t[2]+15 or 
                h_box[3] < t[1]-15 or h_box[1] > t[3]+15):
            return True
    return False

# --- API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/list_videos")
def list_videos():
    return {"videos": [f for f in os.listdir('.') if f.endswith('.mp4')]}

@app.post("/set_source")
async def set_source(data: dict = Body(...)):
    source_val = data.get("source", "3.mp4")
    if str(source_val).isdigit():
        source_val = int(source_val)
    state.update_source(source_val)
    return {"status": "success", "source": str(source_val)}

@app.get("/first_frame")
def get_first_frame():
    with state.lock:
        if not state.cap or not state.cap.isOpened():
            state.cap = cv2.VideoCapture(state.source)
        for _ in range(5): state.cap.grab()
        success, frame = state.cap.read()
    if not success: return {"error": "Source not reachable"}
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.post("/set_roi")
async def set_roi(points: List[Point]):
    state.roi_polygon = np.array([[int(p.x), int(p.y)] for p in points], np.int32)
    return {"status": "success"}

# --- BACKGROUND AI WORKER ---
def process_frames():
    GRACE_PERIOD, COOLDOWN = 1.5, 2.0

    while True:
        with state.lock:
            if state.cap is None or not state.cap.isOpened():
                time.sleep(0.1)
                continue
            success, frame = state.cap.read()

        if not success:
            time.sleep(0.1)
            continue

        if state.roi_polygon is not None:
            results = model.track(frame, persist=True, conf=0.45, verbose=False)
            hands_in_roi, scoopers, pizzas = [], [], []

            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for b, c, track_id in zip(boxes, clss, ids):
                    if is_inside_roi(b, state.roi_polygon):
                        label = model.names[c].lower()
                        if "hand" in label: 
                            hands_in_roi.append((b, track_id))
                        elif any(x in label for x in ["scoop", "spoon", "tool"]): 
                            scoopers.append(b)
                            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 165, 255), 2)
                            cv2.putText(frame, "SPOON/SCOOP", (int(b[0]), int(b[1]-10)), 0, 0.5, (0, 165, 255), 2)
                        elif "pizza" in label: 
                            pizzas.append(b)
                            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 2)
                            cv2.putText(frame, "PIZZA", (int(b[0]), int(b[1]-10)), 0, 0.5, (255, 0, 255), 2)

            curr = time.time()
            for h_box, track_id in hands_in_roi:
                is_safe = is_colliding(h_box, scoopers) or is_colliding(h_box, pizzas)
                if is_safe:
                    state.hand_timers.pop(track_id, None)
                    color, msg = (0, 255, 0), "SAFE (Using Tool)"
                else:
                    if track_id not in state.hand_timers: 
                        state.hand_timers[track_id] = curr
                    
                    elapsed = curr - state.hand_timers[track_id]
                    
                    if elapsed > GRACE_PERIOD:
                        color, msg = (0, 0, 255), f"VIOLATION! {elapsed:.1f}s"
                        if (curr - state.last_violation_time) > COOLDOWN and not state.already_counted.get(track_id):
                            state.total_violations += 1
                            state.last_violation_time = curr
                            state.already_counted[track_id] = True
                    else:
                        color, msg = (0, 255, 255), f"BARE HAND: {elapsed:.1f}s"
                
                cv2.rectangle(frame, (int(h_box[0]), int(h_box[1])), (int(h_box[2]), int(h_box[3])), color, 2)
                cv2.putText(frame, f"ID:{track_id} {msg}", (int(h_box[0]), int(h_box[1]-10)), 0, 0.5, color, 2)

            # Draw UI Overlays
            cv2.polylines(frame, [state.roi_polygon], True, (255, 255, 0), 2)
            cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"VIOLATIONS: {state.total_violations}", (20, 45), 0, 0.8, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        state.latest_frame = buffer.tobytes()

threading.Thread(target=process_frames, daemon=True).start()

@app.get("/video_feed")
def video_feed():
    def stream():
        while True:
            if state.latest_frame:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + state.latest_frame + b'\r\n')
            time.sleep(0.03)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)