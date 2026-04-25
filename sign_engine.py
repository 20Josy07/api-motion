"""
Sign Engine
- Tokeniza texto en palabras/letras
- Busca señas en el diccionario
- Interpola frames entre poses (transiciones suaves)
- Genera la secuencia final de keypoints
"""

import math
import copy
import re
from typing import Optional
from .sign_dictionary import (
    NEUTRAL_POSE, SIGN_DICTIONARY, get_sign, _strip_accents, list_signs
)


# ── Interpolación ─────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _ease_in_out(t: float) -> float:
    """Suaviza la curva de interpolación (cubic ease-in-out)."""
    return t * t * (3 - 2 * t)

def _interp_keypoint(kp_a: dict, kp_b: dict, t: float) -> dict:
    e = _ease_in_out(t)
    return {
        "x": _lerp(kp_a["x"], kp_b["x"], e),
        "y": _lerp(kp_a["y"], kp_b["y"], e),
        "z": _lerp(kp_a["z"], kp_b["z"], e),
    }

def _interp_hand(hand_a: list, hand_b: list, t: float) -> list:
    return [_interp_keypoint(a, b, t) for a, b in zip(hand_a, hand_b)]

def _interp_pose(pose_a: dict, pose_b: dict, t: float) -> dict:
    result = {}
    all_keys = set(pose_a) | set(pose_b)
    for k in all_keys:
        a = pose_a.get(k, {"x": 0.0, "y": 0.0, "z": 0.0})
        b = pose_b.get(k, {"x": 0.0, "y": 0.0, "z": 0.0})
        result[k] = _interp_keypoint(a, b, t)
    return result

def _interp_face(face_a: dict, face_b: dict, t: float) -> dict:
    result = {}
    all_keys = set(face_a) | set(face_b)
    for k in all_keys:
        a = face_a.get(k, {"x": 0.0, "y": 0.0, "z": 0.0})
        b = face_b.get(k, {"x": 0.0, "y": 0.0, "z": 0.0})
        result[k] = _interp_keypoint(a, b, t)
    return result

def _interp_frame(frame_a: dict, frame_b: dict, t: float) -> dict:
    return {
        "right_hand": _interp_hand(frame_a["right_hand"], frame_b["right_hand"], t),
        "left_hand":  _interp_hand(frame_a["left_hand"],  frame_b["left_hand"],  t),
        "pose":       _interp_pose(frame_a["pose"],        frame_b["pose"],       t),
        "face":       _interp_face(frame_a["face"],        frame_b["face"],       t),
    }


def _interpolate_sign_frames(
    sign_frames: list[dict],
    frames_per_sign: int = 30,
    transition_frames: int = 8
) -> list[dict]:
    """
    Expande los keyframes de una seña en frames interpolados.
    - sign_frames: keyframes crudos del diccionario
    - frames_per_sign: frames totales para la seña
    - transition_frames: frames de transición entre keyframes
    """
    if len(sign_frames) == 1:
        return [sign_frames[0]] * frames_per_sign

    result = []
    n = len(sign_frames)
    frames_per_pair = max(2, frames_per_sign // (n - 1))

    for i in range(n - 1):
        fa = sign_frames[i]
        fb = sign_frames[i + 1]
        for f in range(frames_per_pair):
            t = f / (frames_per_pair - 1)
            result.append(_interp_frame(fa, fb, t))

    return result


def _transition_frames(
    frame_a: dict,
    frame_b: dict,
    n: int = 10
) -> list[dict]:
    """Genera N frames de transición entre dos poses."""
    return [_interp_frame(frame_a, frame_b, i / (n - 1)) for i in range(n)]


# ── Tokenizador ───────────────────────────────────────────────────────────────

def _tokenize(text: str, language: str) -> list[str]:
    """
    Convierte texto en tokens (palabras o letras deletreadas).
    Si una palabra no está en el diccionario, la deletrea.
    """
    words = re.split(r"\s+", text.strip().lower())
    tokens = []
    lang_signs = set(list_signs(language))

    for word in words:
        clean = _strip_accents(word.strip(".,!?¿¡"))
        if clean in lang_signs or word in lang_signs:
            tokens.append(clean if clean in lang_signs else word)
        else:
            # Deletrea letra por letra si existe el alfabeto
            for letter in clean:
                if letter in lang_signs:
                    tokens.append(letter)
                # Si no hay seña para esa letra, se omite
    return tokens


# ── Motor principal ───────────────────────────────────────────────────────────

def generate_sign_sequence(
    text: str,
    language: str = "asl",
    fps: int = 30,
    speed: float = 1.0,  # 0.5 = lento, 1.0 = normal, 2.0 = rápido
) -> dict:
    """
    Genera la secuencia completa de frames para animar el avatar.

    Returns:
        {
            "language": str,
            "text": str,
            "tokens": [str],
            "fps": int,
            "total_frames": int,
            "duration_seconds": float,
            "frames": [
                {
                    "frame": int,
                    "time": float,
                    "token": str | None,
                    "right_hand": [...],
                    "left_hand": [...],
                    "pose": {...},
                    "face": {...},
                }
            ]
        }
    """
    tokens = _tokenize(text, language)

    base_sign_frames  = max(8, int(30 / speed))
    base_trans_frames = max(4, int(10 / speed))

    all_frames: list[dict] = []
    frame_labels: list[Optional[str]] = []

    # Pose inicial neutral
    current_pose = copy.deepcopy(NEUTRAL_POSE)

    # Pausa inicial
    pause = 8
    for _ in range(pause):
        all_frames.append(copy.deepcopy(NEUTRAL_POSE))
        frame_labels.append(None)

    for token in tokens:
        sign = get_sign(language, token)
        if sign is None:
            continue

        # Transición desde pose actual hasta primera pose de la seña
        trans = _transition_frames(current_pose, sign[0], base_trans_frames)
        for f in trans:
            all_frames.append(f)
            frame_labels.append(token)

        # Frames de la seña
        expanded = _interpolate_sign_frames(sign, frames_per_sign=base_sign_frames)
        for f in expanded:
            all_frames.append(f)
            frame_labels.append(token)

        current_pose = expanded[-1] if expanded else sign[-1]

        # Breve pausa entre palabras
        pause = max(4, int(8 / speed))
        for _ in range(pause):
            all_frames.append(copy.deepcopy(current_pose))
            frame_labels.append(None)

    # Transición de regreso a neutral
    trans = _transition_frames(current_pose, NEUTRAL_POSE, base_trans_frames)
    for f in trans:
        all_frames.append(f)
        frame_labels.append(None)

    # Construir respuesta
    result_frames = []
    for i, (frame, label) in enumerate(zip(all_frames, frame_labels)):
        result_frames.append({
            "frame": i,
            "time": round(i / fps, 4),
            "token": label,
            **frame
        })

    return {
        "language": language,
        "text": text,
        "tokens": tokens,
        "fps": fps,
        "total_frames": len(result_frames),
        "duration_seconds": round(len(result_frames) / fps, 2),
        "frames": result_frames,
    }
