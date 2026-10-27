"""Runs one image through an organ's model and returns everything the UI needs."""
import json, sys, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
sys.path.insert(0, str(MODELS_DIR))          # so `import model` finds model.py
from model import load_model                 # noqa: E402

ORGANS = ["brain", "skin", "retina"]
_cache = {}                                   # models stay in memory after first load


def get(organ):
    """Load an organ's model and config once, then reuse."""
    if organ not in ORGANS:
        raise ValueError(f"unknown organ '{organ}', expected one of {ORGANS}")
    if organ not in _cache:
        m, classes, params = load_model(MODELS_DIR / f"{organ}_model.pt", device="cpu")
        cfg = json.load(open(MODELS_DIR / f"{organ}_config.json"))
        _cache[organ] = (m, classes, params, cfg)
    return _cache[organ]


def preprocess(img, cfg):
    """Resize keeping proportions, pad with black, normalise. Matches training exactly."""
    size = cfg["image_size"]
    img = img.convert("RGB" if cfg.get("colour") else "L")
    W, H = img.size
    s = size / max(W, H)
    nw, nh = max(1, round(W * s)), max(1, round(H * s))
    canvas = Image.new(img.mode, (size, size), 0 if img.mode == "L" else (0, 0, 0))
    canvas.paste(img.resize((nw, nh), Image.BICUBIC), ((size - nw) // 2, (size - nh) // 2))

    a = np.asarray(canvas, dtype=np.float32) / 255.0
    t = torch.from_numpy(a)
    t = t.permute(2, 0, 1) if cfg.get("colour") else t.unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(cfg["normalise"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["normalise"]["std"]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0), np.asarray(canvas)


def predict(image_path, organ):
    model, classes, params, cfg = get(organ)
    img = Image.open(image_path)
    x, display = preprocess(img, cfg)

    with torch.no_grad():
        logits, p, w, seg = model(x)
    prob = logits.softmax(1)[0].numpy()
    p = p[0].numpy()
    w = w[0].numpy()

    # skin uses a threshold on the malignant class; the others take the top class
    thr = cfg.get("malignant_threshold")
    if thr is not None and prob[0] >= thr:
        idx = 0
    elif thr is not None:
        idx = 1 + int(prob[1:].argmax())
    else:
        idx = int(prob.argmax())

    # the rules that actually fired for this image
    top = np.argsort(-w)[:5]
    by_id = {r["id"]: r for r in cfg["rules"]}
    fired = []
    for r in top:
        rule = by_id.get(f"R{int(r):02d}")
        if rule and w[r] > 0.01:
            fired.append({"id": rule["id"], "text": rule["text"],
                          "conclusion": rule["conclusion"],
                          "strength": round(float(w[r]), 3)})

    # describe each parameter in words, using the fuzzy sets
           # describe each parameter in words, using the fuzzy sets
    if "fuzzy_centres" in cfg:
        centres = np.array(cfg["fuzzy_centres"])
    else:
        centres = np.tile(np.linspace(0, 1, len(cfg["sets"])), (len(params), 1))
    readings = []
    for i, name in enumerate(params):
        label = cfg["sets"][int(np.argmin(np.abs(centres[i] - p[i])))] if centres is not None else ""
        readings.append({"name": name, "value": round(float(p[i]), 3), "level": label})

    out = {
        "organ": organ,
        "prediction": classes[idx],
        "confidence": round(float(prob[idx]), 3),
        "uncertain": bool(prob[idx] < cfg.get("confidence_threshold", 0.5)),
        "all_classes": {c: round(float(prob[i]), 3) for i, c in enumerate(classes)},
        "parameters": readings,
        "rules": fired,
        "has_outline": seg is not None,
    }

    if seg is not None:                       # brain and skin only
        mask = (torch.sigmoid(seg)[0, 0].numpy() > 0.5)
        out["outline_area"] = round(float(mask.mean()), 4)
        out["_mask"] = mask                   # kept for the API to draw later
    out["_display"] = display
    return out


if __name__ == "__main__":
    r = predict(sys.argv[1], sys.argv[2])
    print(f"\norgan: {r['organ']}")
    print(f"prediction: {r['prediction']}  (confidence {r['confidence']})"
          + ("   [UNCERTAIN]" if r["uncertain"] else ""))
    print("\nall classes:", r["all_classes"])
    print("\nparameters:")
    for p in r["parameters"]:
        print(f"  {p['name']:22s} {p['value']:.3f}  {p['level']}")
    print("\nrules that fired:")
    for rr in r["rules"]:
        print(f"  [{rr['strength']:.2f}] {rr['text']}")
    if r["has_outline"]:
        print(f"\nlesion area: {r['outline_area']:.1%} of the image")