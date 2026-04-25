"""
Sign Language Dictionary
Keypoint format:
  - right_hand: 21 puntos (x, y, z) — MediaPipe Hand Landmarks
  - left_hand:  21 puntos (x, y, z)
  - pose:       33 puntos (x, y, z) — MediaPipe Pose
  - face:        6 puntos clave (x, y, z) — boca, cejas, ojos
  
Coordenadas normalizadas: x,y en [-1, 1], z profundidad relativa
"""

import math
from typing import Optional

# ── Pose base (cuerpo en reposo) ──────────────────────────────────────────────
NEUTRAL_POSE = {
    "right_hand": [
        {"x": 0.15, "y": -0.50, "z": 0.0},   # 0 WRIST
        {"x": 0.18, "y": -0.62, "z": 0.0},   # 1 THUMB_CMC
        {"x": 0.22, "y": -0.70, "z": 0.0},   # 2 THUMB_MCP
        {"x": 0.26, "y": -0.76, "z": 0.0},   # 3 THUMB_IP
        {"x": 0.29, "y": -0.82, "z": 0.0},   # 4 THUMB_TIP
        {"x": 0.17, "y": -0.65, "z": 0.0},   # 5 INDEX_MCP
        {"x": 0.17, "y": -0.74, "z": 0.0},   # 6 INDEX_PIP
        {"x": 0.17, "y": -0.82, "z": 0.0},   # 7 INDEX_DIP
        {"x": 0.17, "y": -0.88, "z": 0.0},   # 8 INDEX_TIP
        {"x": 0.15, "y": -0.66, "z": 0.0},   # 9 MIDDLE_MCP
        {"x": 0.15, "y": -0.76, "z": 0.0},   # 10 MIDDLE_PIP
        {"x": 0.15, "y": -0.84, "z": 0.0},   # 11 MIDDLE_DIP
        {"x": 0.15, "y": -0.91, "z": 0.0},   # 12 MIDDLE_TIP
        {"x": 0.13, "y": -0.64, "z": 0.0},   # 13 RING_MCP
        {"x": 0.13, "y": -0.73, "z": 0.0},   # 14 RING_PIP
        {"x": 0.13, "y": -0.80, "z": 0.0},   # 15 RING_DIP
        {"x": 0.13, "y": -0.86, "z": 0.0},   # 16 RING_TIP
        {"x": 0.11, "y": -0.61, "z": 0.0},   # 17 PINKY_MCP
        {"x": 0.11, "y": -0.68, "z": 0.0},   # 18 PINKY_PIP
        {"x": 0.11, "y": -0.73, "z": 0.0},   # 19 PINKY_DIP
        {"x": 0.11, "y": -0.77, "z": 0.0},   # 20 PINKY_TIP
    ],
    "left_hand": [
        {"x": -0.15, "y": -0.50, "z": 0.0},
        {"x": -0.18, "y": -0.62, "z": 0.0},
        {"x": -0.22, "y": -0.70, "z": 0.0},
        {"x": -0.26, "y": -0.76, "z": 0.0},
        {"x": -0.29, "y": -0.82, "z": 0.0},
        {"x": -0.17, "y": -0.65, "z": 0.0},
        {"x": -0.17, "y": -0.74, "z": 0.0},
        {"x": -0.17, "y": -0.82, "z": 0.0},
        {"x": -0.17, "y": -0.88, "z": 0.0},
        {"x": -0.15, "y": -0.66, "z": 0.0},
        {"x": -0.15, "y": -0.76, "z": 0.0},
        {"x": -0.15, "y": -0.84, "z": 0.0},
        {"x": -0.15, "y": -0.91, "z": 0.0},
        {"x": -0.13, "y": -0.64, "z": 0.0},
        {"x": -0.13, "y": -0.73, "z": 0.0},
        {"x": -0.13, "y": -0.80, "z": 0.0},
        {"x": -0.13, "y": -0.86, "z": 0.0},
        {"x": -0.11, "y": -0.61, "z": 0.0},
        {"x": -0.11, "y": -0.68, "z": 0.0},
        {"x": -0.11, "y": -0.73, "z": 0.0},
        {"x": -0.11, "y": -0.77, "z": 0.0},
    ],
    "face": {
        "left_eye":   {"x": -0.07, "y":  0.12, "z": 0.0},
        "right_eye":  {"x":  0.07, "y":  0.12, "z": 0.0},
        "left_brow":  {"x": -0.09, "y":  0.18, "z": 0.0},
        "right_brow": {"x":  0.09, "y":  0.18, "z": 0.0},
        "mouth_l":    {"x": -0.05, "y": -0.05, "z": 0.0},
        "mouth_r":    {"x":  0.05, "y": -0.05, "z": 0.0},
    },
    "pose": {
        "left_shoulder":  {"x": -0.22, "y": -0.20, "z": 0.0},
        "right_shoulder": {"x":  0.22, "y": -0.20, "z": 0.0},
        "left_elbow":     {"x": -0.30, "y": -0.38, "z": 0.0},
        "right_elbow":    {"x":  0.30, "y": -0.38, "z": 0.0},
        "left_wrist":     {"x": -0.15, "y": -0.50, "z": 0.0},
        "right_wrist":    {"x":  0.15, "y": -0.50, "z": 0.0},
        "head":           {"x":  0.00, "y":  0.30, "z": 0.0},
        "neck":           {"x":  0.00, "y":  0.00, "z": 0.0},
    }
}

# ── Helper: copia profunda de pose neutral ────────────────────────────────────
def _neutral():
    import copy
    return copy.deepcopy(NEUTRAL_POSE)


# ── Helpers para construir poses de dedos ─────────────────────────────────────
def _finger_straight(base_x, base_y, base_z=0.0, direction_y=-1, dx=0.0):
    """Genera 4 keypoints para un dedo extendido (MCP→TIP)."""
    return [
        {"x": base_x + dx*0, "y": base_y + direction_y*0.00, "z": base_z},
        {"x": base_x + dx*1, "y": base_y + direction_y*0.09, "z": base_z},
        {"x": base_x + dx*2, "y": base_y + direction_y*0.16, "z": base_z},
        {"x": base_x + dx*3, "y": base_y + direction_y*0.22, "z": base_z},
    ]

def _finger_curled(base_x, base_y, base_z=0.0):
    """Genera 4 keypoints para un dedo curvado (puño)."""
    return [
        {"x": base_x,       "y": base_y,        "z": base_z},
        {"x": base_x+0.02,  "y": base_y-0.06,   "z": base_z+0.04},
        {"x": base_x+0.01,  "y": base_y-0.04,   "z": base_z+0.08},
        {"x": base_x,       "y": base_y-0.02,   "z": base_z+0.09},
    ]


# ── Diccionario de señas ──────────────────────────────────────────────────────
# Estructura por idioma → palabra/letra → lista de frames (cada frame = pose completa)
# Si una seña tiene múltiples frames se interpolan para animar el movimiento.

SIGN_DICTIONARY: dict[str, dict[str, list[dict]]] = {

    # ── ASL ──────────────────────────────────────────────────────────────────
    "asl": {

        # A: puño cerrado, pulgar al lado
        "a": [{
            **_neutral(),
            "right_hand": [
                {"x":  0.15, "y": -0.50, "z": 0.0},  # wrist
                {"x":  0.18, "y": -0.58, "z": 0.0},  # thumb_cmc
                {"x":  0.22, "y": -0.61, "z": 0.0},  # thumb_mcp
                {"x":  0.26, "y": -0.60, "z": 0.0},  # thumb_ip
                {"x":  0.29, "y": -0.58, "z": 0.0},  # thumb_tip
                *[_finger_curled(0.17+i*0.02, -0.62)[j] for i in range(4) for j in range(1)],  # placeholder
                {"x":  0.17, "y": -0.62, "z": 0.0},
                {"x":  0.17, "y": -0.60, "z": 0.06},
                {"x":  0.17, "y": -0.58, "z": 0.10},
                {"x":  0.17, "y": -0.56, "z": 0.11},
                {"x":  0.15, "y": -0.62, "z": 0.0},
                {"x":  0.15, "y": -0.60, "z": 0.06},
                {"x":  0.15, "y": -0.58, "z": 0.10},
                {"x":  0.15, "y": -0.56, "z": 0.11},
                {"x":  0.13, "y": -0.62, "z": 0.0},
                {"x":  0.13, "y": -0.60, "z": 0.06},
                {"x":  0.13, "y": -0.58, "z": 0.10},
                {"x":  0.13, "y": -0.56, "z": 0.11},
                {"x":  0.11, "y": -0.61, "z": 0.0},
                {"x":  0.11, "y": -0.59, "z": 0.05},
                {"x":  0.11, "y": -0.57, "z": 0.08},
                {"x":  0.11, "y": -0.55, "z": 0.09},
            ]
        }],

        # B: mano plana, dedos juntos hacia arriba
        "b": [{
            **_neutral(),
            "right_hand": [
                {"x":  0.15, "y": -0.50, "z": 0.0},   # wrist
                {"x":  0.13, "y": -0.55, "z": 0.04},  # thumb_cmc
                {"x":  0.13, "y": -0.58, "z": 0.06},  # thumb_mcp
                {"x":  0.13, "y": -0.60, "z": 0.07},  # thumb_ip
                {"x":  0.13, "y": -0.61, "z": 0.08},  # thumb_tip
                {"x":  0.18, "y": -0.60, "z": 0.0},
                {"x":  0.18, "y": -0.72, "z": 0.0},
                {"x":  0.18, "y": -0.82, "z": 0.0},
                {"x":  0.18, "y": -0.90, "z": 0.0},
                {"x":  0.16, "y": -0.61, "z": 0.0},
                {"x":  0.16, "y": -0.74, "z": 0.0},
                {"x":  0.16, "y": -0.84, "z": 0.0},
                {"x":  0.16, "y": -0.93, "z": 0.0},
                {"x":  0.14, "y": -0.60, "z": 0.0},
                {"x":  0.14, "y": -0.72, "z": 0.0},
                {"x":  0.14, "y": -0.81, "z": 0.0},
                {"x":  0.14, "y": -0.88, "z": 0.0},
                {"x":  0.12, "y": -0.58, "z": 0.0},
                {"x":  0.12, "y": -0.68, "z": 0.0},
                {"x":  0.12, "y": -0.75, "z": 0.0},
                {"x":  0.12, "y": -0.80, "z": 0.0},
            ]
        }],

        # HELLO: dos frames — mano levantada y agitada
        "hello": [
            {
                **_neutral(),
                "right_hand": NEUTRAL_POSE["right_hand"],
                "pose": {
                    **NEUTRAL_POSE["pose"],
                    "right_elbow": {"x": 0.38, "y": -0.10, "z": 0.0},
                    "right_wrist": {"x": 0.45, "y":  0.15, "z": 0.0},
                }
            },
            {
                **_neutral(),
                "right_hand": NEUTRAL_POSE["right_hand"],
                "pose": {
                    **NEUTRAL_POSE["pose"],
                    "right_elbow": {"x": 0.38, "y": -0.10, "z": 0.0},
                    "right_wrist": {"x": 0.35, "y":  0.15, "z": 0.0},
                }
            },
        ],

        # THANK YOU: mano en barbilla, mueve hacia afuera
        "thank_you": [
            {
                **_neutral(),
                "pose": {
                    **NEUTRAL_POSE["pose"],
                    "right_wrist": {"x": 0.05, "y": -0.08, "z": 0.15},
                }
            },
            {
                **_neutral(),
                "pose": {
                    **NEUTRAL_POSE["pose"],
                    "right_wrist": {"x": 0.20, "y": -0.20, "z": 0.05},
                }
            },
        ],

        # YES: puño asintiendo
        "yes": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.20, "y": -0.40, "z": 0.0}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.20, "y": -0.50, "z": 0.0}}},
        ],

        # NO: dedo índice y medio juntos, movimiento lateral
        "no": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.25, "y": -0.30, "z": 0.0}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.10, "y": -0.30, "z": 0.0}}},
        ],
    },

    # ── LSC (Lengua de Señas Colombiana) ─────────────────────────────────────
    "lsc": {

        # HOLA: mano abierta en la sien, movimiento de saludo
        "hola": [
            {
                **_neutral(),
                "pose": {
                    **NEUTRAL_POSE["pose"],
                    "right_elbow": {"x": 0.30, "y":  0.05, "z": 0.0},
                    "right_wrist": {"x": 0.22, "y":  0.20, "z": 0.0},
                }
            },
            {
                **_neutral(),
                "pose": {
                    **NEUTRAL_POSE["pose"],
                    "right_elbow": {"x": 0.35, "y":  0.05, "z": 0.0},
                    "right_wrist": {"x": 0.40, "y":  0.20, "z": 0.0},
                }
            },
        ],

        # GRACIAS: mano en la boca, se extiende hacia afuera
        "gracias": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.05, "y": -0.05, "z": 0.12}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.22, "y": -0.18, "z": 0.0}}},
        ],

        # SÍ: cabeza y mano afirmando
        "si": [
            {**_neutral(), "face": {**NEUTRAL_POSE["face"], "mouth_l": {"x": -0.05, "y": -0.06, "z": 0.0}, "mouth_r": {"x": 0.05, "y": -0.06, "z": 0.0}}},
            {**_neutral(), "face": {**NEUTRAL_POSE["face"], "mouth_l": {"x": -0.05, "y": -0.04, "z": 0.0}, "mouth_r": {"x": 0.05, "y": -0.04, "z": 0.0}}},
        ],

        # NO
        "no": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.28, "y": -0.28, "z": 0.0}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.08, "y": -0.28, "z": 0.0}}},
        ],

        # POR FAVOR: mano circular en el pecho
        "por_favor": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.10, "y": -0.25, "z": 0.10}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.20, "y": -0.15, "z": 0.10}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.10, "y": -0.10, "z": 0.10}}},
        ],

        # AYUDA: mano izquierda plana, puño derecho encima y sube
        "ayuda": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"],
                "left_wrist": {"x": -0.18, "y": -0.35, "z": 0.0},
                "right_wrist": {"x": 0.0, "y": -0.38, "z": 0.0}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"],
                "left_wrist": {"x": -0.18, "y": -0.35, "z": 0.0},
                "right_wrist": {"x": 0.0, "y": -0.55, "z": 0.0}}},
        ],

        # AGUA
        "agua": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.08, "y": -0.12, "z": 0.08}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.08, "y": -0.08, "z": 0.08}}},
        ],
    },

    # ── LSE (Lengua de Señas Española) ───────────────────────────────────────
    "lse": {

        "hola": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"],
                "right_elbow": {"x": 0.32, "y": 0.0, "z": 0.0},
                "right_wrist": {"x": 0.25, "y": 0.18, "z": 0.0}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"],
                "right_elbow": {"x": 0.32, "y": 0.0, "z": 0.0},
                "right_wrist": {"x": 0.38, "y": 0.18, "z": 0.0}}},
        ],

        "gracias": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.08, "y": -0.02, "z": 0.10}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.25, "y": -0.15, "z": 0.0}}},
        ],

        "por_favor": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.12, "y": -0.22, "z": 0.08}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.18, "y": -0.12, "z": 0.08}}},
        ],

        "si": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.18, "y": -0.42, "z": 0.0}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.18, "y": -0.52, "z": 0.0}}},
        ],

        "no": [
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.26, "y": -0.30, "z": 0.0}}},
            {**_neutral(), "pose": {**NEUTRAL_POSE["pose"], "right_wrist": {"x": 0.10, "y": -0.30, "z": 0.0}}},
        ],
    },
}


def get_sign(language: str, word: str) -> Optional[list[dict]]:
    """Retorna los frames de una seña, o None si no existe."""
    lang_dict = SIGN_DICTIONARY.get(language.lower())
    if not lang_dict:
        return None
    # Busca la palabra exacta, luego sin tildes, luego minúsculas
    key = word.lower().strip()
    return lang_dict.get(key) or lang_dict.get(_strip_accents(key))


def _strip_accents(text: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def list_languages() -> list[str]:
    return list(SIGN_DICTIONARY.keys())


def list_signs(language: str) -> list[str]:
    return list(SIGN_DICTIONARY.get(language.lower(), {}).keys())
