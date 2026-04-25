"""
Sign Language API — FastAPI
Endpoints:
  POST /sign          → JSON con frames de keypoints
  GET  /languages     → idiomas soportados
  GET  /signs/{lang}  → palabras disponibles por idioma
  GET  /health        → estado del servicio
  GET  /avatar        → página HTML con avatar Three.js
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

from .sign_engine import generate_sign_sequence
from .sign_dictionary import list_languages, list_signs

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sign Language API",
    description="Convierte texto en coordenadas de esqueleto para lenguaje de señas (ASL, LSC, LSE).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos ───────────────────────────────────────────────────────────────────

class SignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, example="hola gracias")
    language: str = Field(default="lsc", example="lsc")
    fps: int = Field(default=30, ge=10, le=60)
    speed: float = Field(default=1.0, ge=0.25, le=3.0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/languages")
async def get_languages():
    """Retorna los idiomas de señas disponibles."""
    return {"languages": list_languages()}


@app.get("/signs/{language}")
async def get_signs(language: str):
    """Retorna las palabras/señas disponibles en un idioma."""
    langs = list_languages()
    if language.lower() not in langs:
        raise HTTPException(
            status_code=404,
            detail=f"Idioma '{language}' no encontrado. Disponibles: {langs}"
        )
    return {
        "language": language.lower(),
        "signs": list_signs(language),
        "count": len(list_signs(language)),
    }


@app.post("/sign")
async def sign(request: SignRequest):
    """
    Convierte texto en una secuencia de frames de keypoints para animar un avatar.

    Retorna:
    - frames[]: lista de poses (right_hand, left_hand, pose, face)
    - Cada punto tiene coordenadas normalizadas x, y, z ∈ [-1, 1]
    - right_hand y left_hand: 21 puntos (MediaPipe Hand Landmarks)
    - pose: puntos del cuerpo (shoulders, elbows, wrists, head, neck)
    - face: 6 puntos clave (ojos, cejas, boca)
    """
    langs = list_languages()
    if request.language.lower() not in langs:
        raise HTTPException(
            status_code=400,
            detail=f"Idioma '{request.language}' no soportado. Disponibles: {langs}"
        )

    try:
        result = generate_sign_sequence(
            text=request.text,
            language=request.language.lower(),
            fps=request.fps,
            speed=request.speed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not result["tokens"]:
        raise HTTPException(
            status_code=422,
            detail=f"No se encontraron señas para el texto dado en '{request.language}'. "
                   f"Palabras disponibles: {list_signs(request.language)}"
        )

    return result


@app.get("/avatar", response_class=HTMLResponse)
async def avatar():
    """Página HTML con avatar 3D animado (Three.js) que consume /sign."""
    return HTMLResponse(content=AVATAR_HTML)


# ── HTML Avatar ───────────────────────────────────────────────────────────────

AVATAR_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sign Language Avatar</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f0f14; color: #e0e0e0; font-family: system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; }
  #header { padding: 16px 24px; background: #1a1a24; border-bottom: 1px solid #2a2a3a; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  #header h1 { font-size: 18px; font-weight: 500; color: #a0c4ff; flex: none; }
  #controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; flex: 1; }
  #text-input { flex: 1; min-width: 180px; padding: 8px 12px; background: #252535; border: 1px solid #3a3a50; border-radius: 8px; color: #e0e0e0; font-size: 14px; outline: none; }
  #text-input:focus { border-color: #a0c4ff; }
  select { padding: 8px 10px; background: #252535; border: 1px solid #3a3a50; border-radius: 8px; color: #e0e0e0; font-size: 13px; outline: none; }
  #btn-sign { padding: 8px 20px; background: #4a6cf7; border: none; border-radius: 8px; color: #fff; font-size: 14px; cursor: pointer; font-weight: 500; }
  #btn-sign:hover { background: #5a7cf8; }
  #btn-sign:disabled { background: #333; color: #666; cursor: default; }
  #canvas-wrap { flex: 1; position: relative; }
  canvas { width: 100% !important; height: 100% !important; }
  #status { position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.7); padding: 6px 16px; border-radius: 20px; font-size: 13px; color: #a0c4ff; pointer-events: none; }
  #token-display { position: absolute; top: 16px; left: 50%; transform: translateX(-50%); background: rgba(74,108,247,0.15); border: 1px solid rgba(74,108,247,0.4); padding: 4px 14px; border-radius: 20px; font-size: 14px; font-weight: 500; color: #a0c4ff; pointer-events: none; min-width: 80px; text-align: center; }
</style>
</head>
<body>
<div id="header">
  <h1>✋ Sign Avatar</h1>
  <div id="controls">
    <input id="text-input" type="text" placeholder="Escribe aquí... ej: hola gracias" value="hola gracias">
    <select id="lang-select">
      <option value="lsc">LSC (Colombia)</option>
      <option value="asl">ASL (USA)</option>
      <option value="lse">LSE (España)</option>
    </select>
    <select id="speed-select">
      <option value="0.5">Lento</option>
      <option value="1" selected>Normal</option>
      <option value="2">Rápido</option>
    </select>
    <button id="btn-sign" onclick="fetchAndAnimate()">Animar ▶</button>
  </div>
</div>
<div id="canvas-wrap">
  <canvas id="c"></canvas>
  <div id="token-display">—</div>
  <div id="status">Listo</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── Three.js setup ────────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0f0f14, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0, 0.1, 2.8);
camera.lookAt(0, 0, 0);

// Luces
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(1, 2, 3);
scene.add(dir);

// Resize
function resize() {
  const wrap = document.getElementById('canvas-wrap');
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ── Avatar esqueleto ──────────────────────────────────────────────────────────
const MAT_JOINT = new THREE.MeshPhongMaterial({ color: 0xa0c4ff });
const MAT_BONE  = new THREE.MeshPhongMaterial({ color: 0x4a6cf7, opacity: 0.8, transparent: true });
const MAT_HAND  = new THREE.MeshPhongMaterial({ color: 0x50fa7b, opacity: 0.9, transparent: true });
const MAT_FACE  = new THREE.MeshPhongMaterial({ color: 0xff9580 });

// Cabeza
const head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 12), MAT_FACE);
scene.add(head);

// Joints del cuerpo (diccionario name → mesh)
const poseJoints = {};
const poseNames = ['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','neck'];
for (const n of poseNames) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 8), MAT_JOINT);
  scene.add(m);
  poseJoints[n] = m;
}

// Bones (cilindros entre joints)
const bonePairs = [
  ['left_shoulder','right_shoulder'],
  ['left_shoulder','left_elbow'],
  ['right_shoulder','right_elbow'],
  ['left_elbow','left_wrist'],
  ['right_elbow','right_wrist'],
  ['neck','left_shoulder'],
  ['neck','right_shoulder'],
];
const boneMeshes = {};
for (const [a, b] of bonePairs) {
  const geo = new THREE.CylinderGeometry(0.018, 0.018, 1, 6);
  const m = new THREE.Mesh(geo, MAT_BONE);
  scene.add(m);
  boneMeshes[`${a}_${b}`] = { mesh: m, a, b };
}

// Manos — 21 puntos cada una + líneas de dedos
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],          // pulgar
  [0,5],[5,6],[6,7],[7,8],          // índice
  [0,9],[9,10],[10,11],[11,12],     // medio
  [0,13],[13,14],[14,15],[15,16],   // anular
  [0,17],[17,18],[18,19],[19,20],   // meñique
  [5,9],[9,13],[13,17],             // palma
];

function createHandSkeleton(color) {
  const joints = [];
  for (let i = 0; i < 21; i++) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.015, 6, 6), new THREE.MeshPhongMaterial({ color }));
    scene.add(m);
    joints.push(m);
  }
  const bones = [];
  for (const [a, b] of HAND_CONNECTIONS) {
    const geo = new THREE.CylinderGeometry(0.008, 0.008, 1, 4);
    const m = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color, opacity: 0.7, transparent: true }));
    scene.add(m);
    bones.push({ mesh: m, a, b });
  }
  return { joints, bones };
}

const rightHand = createHandSkeleton(0x50fa7b);
const leftHand  = createHandSkeleton(0xff9580);

// ── Helpers ───────────────────────────────────────────────────────────────────
function kp2vec(kp) { return new THREE.Vector3(kp.x, kp.y, kp.z); }

function updateBoneBetween(mesh, posA, posB) {
  const a = kp2vec(posA), b = kp2vec(posB);
  const mid = a.clone().add(b).multiplyScalar(0.5);
  const dist = a.distanceTo(b);
  mesh.position.copy(mid);
  mesh.scale.y = dist;
  mesh.lookAt(b);
  mesh.rotateX(Math.PI / 2);
}

function applyFrame(frame) {
  if (!frame) return;

  // Pose
  const p = frame.pose;
  if (p) {
    for (const n of poseNames) {
      if (p[n]) poseJoints[n].position.copy(kp2vec(p[n]));
    }
    if (p.head) head.position.copy(kp2vec(p.head));

    // Actualizar bones
    for (const [key, { mesh, a, b }] of Object.entries(boneMeshes)) {
      if (p[a] && p[b]) updateBoneBetween(mesh, p[a], p[b]);
    }
  }

  // Manos
  function applyHand(hand, kps) {
    if (!kps) return;
    for (let i = 0; i < 21; i++) {
      if (kps[i]) hand.joints[i].position.copy(kp2vec(kps[i]));
    }
    for (const { mesh, a, b } of hand.bones) {
      if (kps[a] && kps[b]) updateBoneBetween(mesh, kps[a], kps[b]);
    }
  }
  applyHand(rightHand, frame.right_hand);
  applyHand(leftHand,  frame.left_hand);
}

// ── Animación ─────────────────────────────────────────────────────────────────
let frames = [];
let currentFrame = 0;
let playing = false;
let lastTime = 0;
let fps = 30;
const statusEl = document.getElementById('status');
const tokenEl  = document.getElementById('token-display');

function setStatus(msg) { statusEl.textContent = msg; }

function animLoop(now) {
  requestAnimationFrame(animLoop);
  if (playing && frames.length > 0) {
    const elapsed = now - lastTime;
    if (elapsed >= 1000 / fps) {
      lastTime = now;
      applyFrame(frames[currentFrame]);
      tokenEl.textContent = frames[currentFrame].token || '—';
      currentFrame++;
      if (currentFrame >= frames.length) {
        currentFrame = 0; // loop
      }
    }
  }
  renderer.render(scene, camera);
}
requestAnimationFrame(animLoop);

// ── Fetch y animar ────────────────────────────────────────────────────────────
async function fetchAndAnimate() {
  const text  = document.getElementById('text-input').value.trim();
  const lang  = document.getElementById('lang-select').value;
  const speed = parseFloat(document.getElementById('speed-select').value);
  const btn   = document.getElementById('btn-sign');

  if (!text) return;

  btn.disabled = true;
  playing = false;
  setStatus('Cargando señas...');
  tokenEl.textContent = '...';

  try {
    const res = await fetch('/sign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language: lang, fps: 30, speed }),
    });

    if (!res.ok) {
      const err = await res.json();
      setStatus('Error: ' + (err.detail || res.statusText));
      return;
    }

    const data = await res.json();
    frames = data.frames;
    fps = data.fps;
    currentFrame = 0;
    playing = true;
    lastTime = performance.now();
    setStatus(`▶ ${data.tokens.join(' · ')} — ${data.duration_seconds}s · ${data.total_frames} frames`);
  } catch (e) {
    setStatus('Error de red: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

// También Enter en el input
document.getElementById('text-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') fetchAndAnimate();
});

// Cargar automáticamente al inicio
window.addEventListener('load', () => fetchAndAnimate());
</script>
</body>
</html>
"""
