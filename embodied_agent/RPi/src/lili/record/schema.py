"""Format sesji. Opis: embodied_agent/doc/v4_design.md, sekcja "Format nagrania"."""

SCHEMA_VERSION = 1

META_FILE = "meta.json"
STATE_FILE = "state.jsonl"
FRAMES_DIR = "frames"

# Klucze w frames/chunk_NNNN.npz
FRAMES = "frames"  # (N, H, W) uint8
T = "t"  # (N,) int64, time.monotonic_ns() po odebraniu klatki
T_SENSOR = "t_sensor"  # (N,) int64, SensorTimestamp z Picamera2
INDEX = "index"  # (N,) int64, numer klatki w sesji

PHASE_MOVE = "move"
PHASE_SETTLE = "settle"
PHASE_LOOK = "look"
PHASE_END = "end"


def chunk_name(n):
    return f"chunk_{n:04d}.npz"


# Zdarzenia w state.jsonl, wszystkie z t = time.monotonic_ns()
def cmd_event(t, pan, tilt, pan_byte, tilt_byte):
    return {"t": t, "type": "cmd", "pan": pan, "tilt": tilt, "pan_byte": pan_byte, "tilt_byte": tilt_byte}


def stm_event(t, channel, position):
    return {"t": t, "type": "stm", "channel": channel, "position": position}


def phase_event(t, phase, step):
    return {"t": t, "type": "phase", "phase": phase, "step": step}


def base_event(t, offset):
    """Deklarowane (nie zmierzone) położenie podstawy robota [m], np. po ręcznym przesunięciu."""
    return {"t": t, "type": "base", "offset": list(offset)}
