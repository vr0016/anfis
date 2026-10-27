"""The functions the LLM can call. It cannot diagnose; it can only read these results."""
import json, sys
from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import predict as _predict, get, ORGANS      # noqa: E402

_knowledge = {o: json.load(open(ROOT / "knowledge" / f"{o}.json")) for o in ORGANS}
      # the most recent analysis, so follow-up questions can refer to it


def analyse_scan(image_path: str, organ: str) -> dict:
    """Run the model. Returns prediction, parameters and the rules that fired."""
    r = _predict(image_path, organ)
    r.pop("_display", None); r.pop("_mask", None)
    
    return r


def get_condition_info(organ: str, condition: str) -> dict:
    """Look up what a condition is and what the usual next steps are."""
    entry = _knowledge.get(organ, {}).get(condition)
    if not entry:
        return {"error": f"no entry for '{condition}' under {organ}",
                "available": list(_knowledge.get(organ, {}))}
    return entry


def list_rules(organ: str, limit: int = 5) -> dict:
    """The fuzzy rules this organ's model uses, most-used first."""
    _, _, _, cfg = get(organ)
    return {"organ": organ,
            "rules": [{"id": r["id"], "text": r["text"],
                       "average_use": r["average_use"]} for r in cfg["rules"][:limit]]}


def model_performance(organ: str) -> dict:
    """How accurate this model is on its test set, and its known limitations."""
    _, classes, params, cfg = get(organ)
    return {"organ": organ, "classes": classes, "parameters": params,
            "metrics": cfg.get("test_metrics", {}),
            "notes": cfg.get("notes", ""),
            "has_outline": cfg.get("has_segmentation", True)}

def web_lookup(question: str) -> dict:
    """Search the web for general medical information this system does not hold."""
    import os
    from google import genai
    from google.genai import types as gtypes

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"error": "web search unavailable: no API key"}
    try:
        client = genai.Client(api_key=key,
                              http_options=gtypes.HttpOptions(timeout=45_000))
        r = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=question,
            config=gtypes.GenerateContentConfig(
                tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())],
                system_instruction=(
                    "Answer the question using web search. Be factual and brief. "
                    "Prefer health-service and medical sources. "
                    "Do not give personal medical advice or suggest treatment for an "
                    "individual; describe general information only. "
                    "Name the sources you relied on."),
            ),
        )
        text = "".join(p.text for p in (r.candidates[0].content.parts or []) if p.text)

        sources = []
        meta = getattr(r.candidates[0], "grounding_metadata", None)
        for c in (getattr(meta, "grounding_chunks", None) or []):
            w = getattr(c, "web", None)
            if w and getattr(w, "uri", None):
                sources.append({"title": getattr(w, "title", ""), "url": w.uri})

        return {"answer": text, "sources": sources[:5]}
    except Exception as e:
        return {"error": f"web search failed: {e}"}


REGISTRY = {f.__name__: f for f in
            [analyse_scan, get_condition_info, list_rules, model_performance, web_lookup]}
SCHEMAS = [
    {"name": "analyse_scan",
     "description": "Run the diagnosis model on an uploaded image. Use once per uploaded scan.",
     "parameters": {"type": "object", "properties": {
         "image_path": {"type": "string", "description": "path to the image file"},
         "organ": {"type": "string", "enum": ORGANS}},
         "required": ["image_path", "organ"]}},
    {"name": "get_condition_info",
     "description": "Look up what a condition is and the usual next steps. Always call this before describing a condition.",
     "parameters": {"type": "object", "properties": {
         "organ": {"type": "string", "enum": ORGANS},
         "condition": {"type": "string", "description": "a class name returned by analyse_scan"}},
         "required": ["organ", "condition"]}},
    {"name": "list_rules",
     "description": "The fuzzy IF-THEN rules a model uses. Use when asked how the system decides.",
     "parameters": {"type": "object", "properties": {
         "organ": {"type": "string", "enum": ORGANS},
         "limit": {"type": "integer"}},
         "required": ["organ"]}},
        {"name": "web_lookup",
     "description": ("Search the web for general medical or scientific information this "
                     "system does not hold: background on a condition, how common it is, "
                     "what a term means, what research says. Use ONLY after checking "
                     "get_condition_info. Never use it to ask what an individual should do."),
     "parameters": {"type": "object", "properties": {
         "question": {"type": "string", "description": "a clear, self-contained question"}},
         "required": ["question"]}},

    {"name": "model_performance",
     "description": "Accuracy and known limitations of a model. Use when asked how reliable it is.",
     "parameters": {"type": "object", "properties": {
         "organ": {"type": "string", "enum": ORGANS}},
         "required": ["organ"]}},
]