# Sign Language API 🤟

API REST que convierte texto en coordenadas de esqueleto para lenguaje de señas, con avatar 3D animado.

## Instalación

```bash
cd sign_api
pip install -r requirements.txt
```

## Correr la API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET`  | `/health` | Estado del servicio |
| `GET`  | `/languages` | Idiomas soportados |
| `GET`  | `/signs/{lang}` | Palabras disponibles |
| `POST` | `/sign` | Texto → JSON keypoints |
| `GET`  | `/avatar` | Avatar 3D en el navegador |

## Uso

### Avatar visual
Abre en el navegador: http://localhost:8000/avatar

### API
```bash
curl -X POST http://localhost:8000/sign \
  -H "Content-Type: application/json" \
  -d '{"text": "hola gracias", "language": "lsc", "fps": 30, "speed": 1.0}'
```

### Respuesta JSON
```json
{
  "language": "lsc",
  "text": "hola gracias",
  "tokens": ["hola", "gracias"],
  "fps": 30,
  "total_frames": 120,
  "duration_seconds": 4.0,
  "frames": [
    {
      "frame": 0,
      "time": 0.0,
      "token": null,
      "right_hand": [
        {"x": 0.15, "y": -0.50, "z": 0.0},
        ...21 puntos MediaPipe...
      ],
      "left_hand": [...21 puntos...],
      "pose": {
        "left_shoulder":  {"x": -0.22, "y": -0.20, "z": 0.0},
        "right_shoulder": {"x":  0.22, "y": -0.20, "z": 0.0},
        "left_elbow":     {"x": -0.30, "y": -0.38, "z": 0.0},
        "right_elbow":    {"x":  0.30, "y": -0.38, "z": 0.0},
        "left_wrist":     {"x": -0.15, "y": -0.50, "z": 0.0},
        "right_wrist":    {"x":  0.15, "y": -0.50, "z": 0.0},
        "head":           {"x":  0.00, "y":  0.30, "z": 0.0},
        "neck":           {"x":  0.00, "y":  0.00, "z": 0.0}
      },
      "face": {
        "left_eye":   {"x": -0.07, "y": 0.12, "z": 0.0},
        "right_eye":  {"x":  0.07, "y": 0.12, "z": 0.0},
        "mouth_l":    {"x": -0.05, "y": -0.05, "z": 0.0},
        "mouth_r":    {"x":  0.05, "y": -0.05, "z": 0.0},
        ...
      }
    }
  ]
}
```

## Idiomas soportados

| Código | Idioma |
|--------|--------|
| `lsc`  | Lengua de Señas Colombiana |
| `asl`  | American Sign Language |
| `lse`  | Lengua de Señas Española |

## Palabras disponibles

### LSC
`hola`, `gracias`, `si`, `no`, `por_favor`, `ayuda`, `agua`

### ASL
`a`, `b`, `hello`, `thank_you`, `yes`, `no`

### LSE
`hola`, `gracias`, `por_favor`, `si`, `no`

## Parámetros de /sign

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `text` | string | — | Texto a convertir |
| `language` | string | `lsc` | Idioma (`lsc`, `asl`, `lse`) |
| `fps` | int | `30` | Frames por segundo |
| `speed` | float | `1.0` | Velocidad (0.25–3.0) |

## Estructura de keypoints

Basado en MediaPipe:
- **right_hand / left_hand**: 21 puntos (muñeca + 5 dedos × 4 joints)
- **pose**: 8 puntos del cuerpo (hombros, codos, muñecas, cuello, cabeza)
- **face**: 6 puntos (ojos, cejas, boca)

Coordenadas normalizadas: x, y, z ∈ [-1.0, 1.0]

## Extender el diccionario

Edita `app/sign_dictionary.py` y agrega señas al diccionario `SIGN_DICTIONARY`:

```python
SIGN_DICTIONARY["lsc"]["nueva_seña"] = [
    # Frame 1 (inicio)
    {
        **_neutral(),
        "pose": {
            **NEUTRAL_POSE["pose"],
            "right_wrist": {"x": 0.20, "y": -0.30, "z": 0.0},
        }
    },
    # Frame 2 (fin del movimiento)
    {
        **_neutral(),
        "pose": {
            **NEUTRAL_POSE["pose"],
            "right_wrist": {"x": 0.35, "y": -0.10, "z": 0.0},
        }
    },
]
```

## Documentación automática

- Swagger UI: http://localhost:8000/docs
- ReDoc:       http://localhost:8000/redoc
