"""Lyre Tutor web service (FastAPI + LangGraph agent).

Endpoints
  GET  /health   -> {"status": "ok"}
  POST /agent    -> JSON {"chat_id": ..., "text": "..."}            -> {"reply": "..."}
  POST /listen   -> multipart: chat_id (field) + file (audio/video) -> {"reply": "..."}

Environment variables
  GEMINI_API_KEY  required  your Gemini key
  API_SECRET      required  shared secret; callers send it in the X-API-Key header
  MODEL_NAMES     optional  comma-separated models tried in order
                            (default: gemini-3.5-flash-lite,gemini-3.1-flash-lite)
  MODEL_NAME      optional  single model (used if MODEL_NAMES is not set)
"""
import os
import secrets
import tempfile
from typing import Union

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# ---------------------------------------------------------------- config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
# Models are tried in order; if one is busy or out of quota, the next one is used.
MODEL_NAMES = [
    m.strip()
    for m in os.environ.get(
        "MODEL_NAMES",
        os.environ.get("MODEL_NAME", "gemini-3.5-flash-lite,gemini-3.1-flash-lite"),
    ).split(",")
    if m.strip()
]
MODEL_NAME = MODEL_NAMES[0]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

if not GEMINI_API_KEY or not API_SECRET:
    raise RuntimeError("Set GEMINI_API_KEY and API_SECRET before starting the service.")

# ---------------------------------------------------------------- lyre logic
# 19 strings: white keys from F3 (MIDI 53) to C6 (MIDI 84)
WHITE = {0, 2, 4, 5, 7, 9, 11}
LYRE = [m for m in range(53, 85) if m % 12 in WHITE]
STRING_OF = {m: i + 1 for i, m in enumerate(LYRE)}
STRING_HZ = {i + 1: float(librosa.midi_to_hz(m)) for i, m in enumerate(LYRE)}
LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Public-domain / traditional songs only
SONGS = {
    "twinkle twinkle": "C4 C4 G4 G4 A4 A4 G4 F4 F4 E4 E4 D4 D4 C4",
    "ode to joy": "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4 D4 D4",
    "mary had a little lamb": "E4 D4 C4 D4 E4 E4 E4 D4 D4 D4 E4 G4 G4",
    "happy birthday": "G4 G4 A4 G4 C5 B4 G4 G4 A4 G4 D5 C5",
    "frere jacques": "C4 D4 E4 C4 C4 D4 E4 C4 E4 F4 G4 E4 F4 G4",
}


def to_midi(note: str) -> int:
    letter, rest, acc = note[0].upper(), note[1:], 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + LETTER[letter] + acc


def convert(song: str):
    """Return (shift_in_semitones, [string numbers]) or (None, None)."""
    midis = [to_midi(n) for n in song.split()]
    for shift in sorted(range(-24, 25), key=abs):
        shifted = [m + shift for m in midis]
        if all(m in STRING_OF for m in shifted):
            return shift, [STRING_OF[m] for m in shifted]
    return None, None


def explain(song: str) -> str:
    try:
        shift, strings = convert(song)
    except (KeyError, ValueError, IndexError):
        return "Sorry, I could not read those notes. Use names like C4 D4 E4."
    if strings is None:
        return "Sorry, this song does not fit your lyre."
    text = "Play strings: " + " - ".join(map(str, strings))
    if shift != 0:
        direction = "up" if shift > 0 else "down"
        text += f"\n(I moved the song {abs(shift)} semitone(s) {direction} to fit your lyre.)"
    return text


def nearest_string(hz: float, max_cents: int = 60):
    best = min(STRING_HZ, key=lambda s: abs(1200 * np.log2(hz / STRING_HZ[s])))
    cents = 1200 * np.log2(hz / STRING_HZ[best])
    return (best if abs(cents) <= max_cents else None), round(float(cents))


def detect_strings_v2(path: str):
    """Return a list of (string_or_None, cents_off) for each plucked note."""
    y, sr = librosa.load(path, sr=22050)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
    results = []
    for i, start in enumerate(onsets):
        if start / sr < 0.3:  # ignore the click when recording starts
            continue
        end = onsets[i + 1] if i + 1 < len(onsets) else len(y)
        seg = y[start + int(0.1 * sr): min(end, start + int(0.8 * sr))]
        if len(seg) < 2048:
            continue
        f0, _, _ = librosa.pyin(
            seg, sr=sr,
            fmin=librosa.note_to_hz("C3"), fmax=librosa.note_to_hz("C7"),
        )
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            continue
        results.append(nearest_string(float(np.median(f0))))
    return results


# ---------------------------------------------------------------- agent tools
LAST_SONG = {}      # chat_id -> expected string numbers
PENDING_AUDIO = {}  # chat_id -> path of the recording waiting to be checked


def _chat_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


@tool
def list_songs() -> str:
    """List the songs the tutor can teach. Use it when the user asks for suggestions or what is available."""
    return "Available songs: " + ", ".join(sorted(SONGS))


@tool
def find_song(name: str, config: RunnableConfig) -> str:
    """Look up a song by name and return the lyre string numbers to play."""
    notes = SONGS.get(name.lower().strip())
    if notes is None:
        return "I don't know this song yet."
    _, strings = convert(notes)
    if strings:
        LAST_SONG[_chat_id(config)] = strings
    return explain(notes)


@tool
def convert_notes(notes: str) -> str:
    """Convert note names such as 'C4 D4 E4' into lyre string numbers."""
    return explain(notes)


@tool
def check_recording(config: RunnableConfig) -> str:
    """Listen to the user's latest recording and compare it with the last song taught."""
    chat_id = _chat_id(config)
    expected = LAST_SONG.get(chat_id)
    path = PENDING_AUDIO.get(chat_id)
    if not path or not os.path.exists(path):
        return "There is no recording to check. Ask the user to send one."
    if not expected:
        return "No song has been taught yet. Ask which song the user wants to practise."
    try:
        heard = detect_strings_v2(path)
    except Exception:
        return "The recording could not be read. Ask the user to record again."
    if not heard:
        return "I could not hear any notes. Ask the user to record again, closer and in a quiet room."
    issues = []
    for i, (s, cents) in enumerate(heard):
        if i >= len(expected):
            issues.append(f"Note {i+1}: the song has no more notes, but an extra note was heard.")
            break
        if s != expected[i]:
            if s is None:
                issues.append(
                    f"Note {i+1}: expected string {expected[i]}, but the pitch heard matches no string. "
                    f"The string is probably out of tune ({cents:+d} cents from the nearest string). "
                    "Ask the user to tune it."
                )
            else:
                issues.append(f"Note {i+1}: expected string {expected[i]}, heard string {s} ({cents:+d} cents)")
        elif abs(cents) > 35:
            issues.append(
                f"Note {i+1}: string {s} is right but sounds {'flat' if cents < 0 else 'sharp'} "
                f"({cents:+d} cents), so the string may need tuning"
            )
    if not issues:
        return f"All {len(heard)} notes heard are correct (compared with the start of the song)."
    return "\n".join(issues)


SYSTEM_PROMPT = """You are Lyre Tutor, a friendly assistant that teaches beginners to play a 19-string lyre.
The lyre is tuned to the white keys from F3 (string 1) to C6 (string 19). Users know no music theory.

Rules:
1. NEVER work out string numbers yourself. Always use the tools.
2. If the user asks for suggestions or which songs you know, call list_songs. Only suggest songs from that list.
3. If the user asks for a song, call find_song. If the song is unknown, say so honestly and never invent notes.
4. If the user gives notes like 'C4 D4 E4', call convert_notes.
5. Reply in the user's language (Armenian, Russian or English), in short simple sentences, without music jargon.
6. Show the string numbers clearly, then give one beginner tip (for example: play slowly, one phrase at a time).
7. If a tool says the song does not fit the lyre, tell the user that instead of guessing.
8. If the user sends a recording, call check_recording and explain the result kindly. Say which notes were wrong and what to try. Never say a recording is correct unless the tool says so."""

CHECKPOINTER = MemorySaver()  # shared, so chat memory survives a model switch
_AGENTS = {}


def get_agent(model_name: str):
    if model_name not in _AGENTS:
        llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=GEMINI_API_KEY, max_retries=1, timeout=30
        )
        _AGENTS[model_name] = create_agent(
            model=llm,
            tools=[list_songs, find_song, convert_notes, check_recording],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=CHECKPOINTER,
        )
    return _AGENTS[model_name]


def ask(text: str, chat_id: str) -> str:
    last_error = None
    for model_name in MODEL_NAMES:
        try:
            result = get_agent(model_name).invoke(
                {"messages": [("user", text)]},
                config={"configurable": {"thread_id": str(chat_id)}},
            )
        except Exception as exc:  # busy (503), out of quota (429), timeout...
            print(f"model {model_name} failed: {type(exc).__name__}: {str(exc)[:150]}")
            last_error = exc
            continue
        content = result["messages"][-1].content
        if isinstance(content, str):
            return content
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    raise last_error


# ---------------------------------------------------------------- web API
app = FastAPI(title="Lyre Tutor")


def require_key(x_api_key: str):
    if not secrets.compare_digest(x_api_key or "", API_SECRET):
        raise HTTPException(status_code=401, detail="Invalid API key")


class AgentRequest(BaseModel):
    chat_id: Union[int, str]
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agent")
def agent_endpoint(req: AgentRequest, x_api_key: str = Header(default="")):
    require_key(x_api_key)
    try:
        return {"reply": ask(req.text, str(req.chat_id))}
    except Exception as exc:
        print("agent error:", repr(exc))
        raise HTTPException(status_code=503, detail="The tutor is unavailable right now. Try again in a minute.")


@app.post("/listen")
def listen_endpoint(
    chat_id: str = Form(...),
    file: UploadFile = File(...),
    x_api_key: str = Header(default=""),
):
    require_key(x_api_key)
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    PENDING_AUDIO[chat_id] = path
    try:
        reply = ask("I sent a recording of me playing. Please check it.", chat_id)
        return {"reply": reply}
    except Exception as exc:
        print("listen error:", repr(exc))
        raise HTTPException(status_code=503, detail="The tutor is unavailable right now. Try again in a minute.")
    finally:
        PENDING_AUDIO.pop(chat_id, None)
        if os.path.exists(path):
            os.remove(path)
