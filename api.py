
"""HTTP interface. Run with: python -m uvicorn backend.api:app --reload"""
import base64, io, sys, uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import predict, get, ORGANS  # noqa: E402

app = FastAPI(title="Clinical Diagnosis Support", version="0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS = Path(__file__).parent / "_uploads"
UPLOADS.mkdir(exist_ok=True)
_sessions = {}  # session_id -> Agent

def overlay(display, mask):
    rgb = np.stack([display] * 3, -1) if display.ndim == 2 else display.copy()
    rgb = rgb.astype(np.float32)
    rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.array([0.0, 200.0, 220.0])
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@app.get("/organs")
def organs():
    out = []
    for o in ORGANS:
        _, classes, params, cfg = get(o)
        out.append({
            "organ": o,
            "classes": classes,
            "class_full": cfg.get("class_full", {}),
            "parameters": params,
            "has_outline": cfg.get("has_segmentation", True),
            "metrics": cfg.get("test_metrics", {}),
        })
    return {"organs": out}

@app.post("/predict")
async def run(file: UploadFile = File(...), organ: str = Form(...)):
    """Analyse one image. Returns the result and a session id for follow-up chat."""
    if organ not in ORGANS:
        raise HTTPException(400, f"unknown organ '{organ}'; expected {ORGANS}")

    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "could not read that file as an image")

    sid = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{sid}.png"
    img.save(path, format="PNG")

    r = predict(path, organ)
    display, mask = r.pop("_display"), r.pop("_mask", None)

    if mask is not None:
        r["overlay"] = overlay(display, mask)
        _sessions.pop(sid, None)  # fresh agent for a fresh scan

    globals().setdefault("_organs", {})[sid] = organ
    r["session_id"] = sid
    r["disclaimer"] = (
        "Decision support only. This is not a medical diagnosis. "
        "Please consult a qualified clinician."
    )
    return r

class ChatIn(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(body: ChatIn):
    """Ask the assistant about the scan analysed in this session."""
    path = UPLOADS / f"{body.session_id}.png"

    if not path.exists():
        raise HTTPException(
            404,
            "no scan found for that session; upload one first",
        )

    if body.session_id not in _sessions:
        try:
            from agent import Agent
            _sessions[body.session_id] = Agent()
        except Exception as e:
            raise HTTPException(503, f"assistant unavailable: {e}")

    agent = _sessions[body.session_id]

    # tell the agent where the image is, so it can call analyse_scan if it needs to
    organ = globals().get("_organs", {}).get(body.session_id)

    if not organ:
        raise HTTPException(404, "session has no organ recorded; upload again")

    msg = (
        f"(context: the user uploaded a {organ} scan, saved at '{path}'. "
        f"Use organ='{organ}' for any tool call. Do not use any other organ.)\n\n"
        f"{body.message}"
    )

    try:
        return {"reply": agent.ask(msg)}
    except Exception as e:
        raise HTTPException(503, f"assistant error: {e}")

@app.delete("/session/{sid}")
def end_session(sid: str):
    _sessions.pop(sid, None)
    (UPLOADS / f"{sid}.png").unlink(missing_ok=True)
    return {"ended": sid}

@app.get("/")
def root():
    return {"status": "ok", "organs": ORGANS, "docs": "/docs"}
