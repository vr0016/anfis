"""The LLM layer. It explains model output; it never diagnoses on its own."""
import json, os, sys, time
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools  # noqa: E402

MODEL = "gemini-3.5-flash-lite"

SYSTEM = """You help a clinician or patient understand the output of a medical image
analysis system. You are NOT the diagnostic system; three trained models are, and you
read their results through tools.

Rules you must follow:
- Never state a finding the tools did not return. If you do not have it, say so.
- - Always call get_condition_info before describing a condition or its next steps.
  Use only what that returns. Never add treatment advice of your own.
- That tool also returns a "self_care" list of practical things the person can do
  themselves. Include these when they ask what they can do, or when giving a full
  explanation. They are general guidance from health services, not prescriptions.
- For questions beyond the models and the condition files, use web_lookup. Always try
  get_condition_info first; web_lookup is for background, statistics, terminology and
  research, not for what this particular person should do.
- When you use web_lookup, say the information came from a web search and name the
  sources it returned.  
- Report the confidence the model gave. If "uncertain" is true, say plainly that the
  model was not confident.
- Explain the reasoning using the parameters and the rules that fired, in plain words.
- Always end by saying this is decision support, not a diagnosis, and a clinician
  should confirm.
- If asked something the tools cannot answer, say you do not have that information.
- Keep answers short. Do not repeat the raw numbers back; explain what they mean."""


class Agent:
    def __init__(self, api_key=None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set in this shell")
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=60_000),
        )
        self.history = []
        self.cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=tools.SCHEMAS)],
        )

    def ask(self, message, verbose=False):
        self.history.append(types.Content(role="user",
                                          parts=[types.Part(text=message)]))
        for step in range(6):
            if verbose:
                print(f"   [thinking... step {step+1}]", flush=True)

            r = None
            for attempt in range(4):
                try:
                    r = self.client.models.generate_content(
                        model=MODEL, contents=self.history, config=self.cfg)
                    break
                except Exception as e:
                    if any(s in str(e) for s in ("503", "504", "UNAVAILABLE", "DEADLINE")):
                        wait = 2 ** attempt
                        if verbose:
                            print(f"   [busy, retrying in {wait}s]", flush=True)
                        time.sleep(wait)
                        continue
                    return f"The request failed: {type(e).__name__}: {e}"
            if r is None:
                return "The model is busy right now. Try again in a minute."

            cand = r.candidates[0].content
            self.history.append(cand)

            calls = [p.function_call for p in (cand.parts or []) if p.function_call]
            if not calls:
                return "".join(p.text for p in (cand.parts or []) if p.text)

            replies = []
            for c in calls:
                if verbose:
                    print(f"   [tool] {c.name}({dict(c.args)})", flush=True)
                try:
                    out = tools.REGISTRY[c.name](**dict(c.args))
                except Exception as e:
                    out = {"error": str(e)}
                replies.append(types.Part.from_function_response(
                    name=c.name, response={"result": out}))
            self.history.append(types.Content(role="user", parts=replies))
        return "I could not complete that request."


if __name__ == "__main__":
    a = Agent()
    print("Agent ready. Type a question, or 'quit'.\n")
    print("Try: analyse samples/brain_glioma_1.jpg as a brain scan\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            print("\n" + a.ask(q, verbose=True) + "\n")