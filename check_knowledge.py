import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
from model import load_model

ok = True
for organ in ["brain", "skin", "retina"]:
    _, classes, _ = load_model(ROOT / "models" / f"{organ}_model.pt", device="cpu")
    k = json.load(open(ROOT / "knowledge" / f"{organ}.json"))
    missing = [c for c in classes if c not in k]
    extra = [c for c in k if c not in classes]
    print(f"{organ:8s} classes={classes}")
    if missing: print(f"   MISSING from knowledge file: {missing}"); ok = False
    if extra:   print(f"   EXTRA (not a model class): {extra}"); ok = False
    for c in classes:
        if c in k:
            for field in ["name", "what_it_is", "typical_next_steps", "urgency", "source"]:
                if not k[c].get(field):
                    print(f"   {c}: empty field '{field}'"); ok = False
print("\nall good" if ok else "\nfix the issues above")