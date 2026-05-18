#!/usr/bin/env python3
"""Gradio + Three.js PBR 椅子查看器 — 浏览器端实时渲染，鼠标拖拽旋转"""

import gradio as gr
import os
import json

DESKTOP = "/home/ai/Desktop"

# 扫描 cubemap 目录
env_dirs = {}
for d in sorted(os.listdir(DESKTOP)):
    full = os.path.join(DESKTOP, d)
    if os.path.isdir(full) and os.path.exists(os.path.join(full, "posx.jpg")):
        env_dirs[d] = full

env_list = list(env_dirs.keys())
default_env = "Regal_Palace_Ballroom_cubemap" if "Regal_Palace_Ballroom_cubemap" in env_list else env_list[0]

# 构建环境贴图路径 JSON — 使用 /assets/ 前缀（将通过 FastAPI StaticFiles 挂载）
env_paths = {}
for name, path in env_dirs.items():
    env_paths[name] = {
        "px": f"/assets/{name}/posx.jpg",
        "nx": f"/assets/{name}/negx.jpg",
        "py": f"/assets/{name}/posy.jpg",
        "ny": f"/assets/{name}/negy.jpg",
        "pz": f"/assets/{name}/posz.jpg",
        "nz": f"/assets/{name}/negz.jpg",
    }

VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#1a1a1a; overflow:hidden; font-family:system-ui,sans-serif; }
  #canvas { width:100vw; height:100vh; display:block; }
  #panel {
    position:fixed; top:12px; right:12px; background:rgba(30,30,30,0.92);
    padding:16px; border-radius:10px; color:#ddd; min-width:220px;
    backdrop-filter:blur(8px); box-shadow:0 4px 20px rgba(0,0,0,0.5);
  }
  #panel label { display:block; margin-top:10px; font-size:13px; }
  #panel input[type=range] { width:100%; margin:4px 0; }
  #panel select { width:100%; padding:4px; margin:4px 0; background:#333; color:#ddd; border:1px solid #555; border-radius:4px; }
  #panel .val { float:right; color:#8cf; font-size:12px; }
  #panel h3 { margin:0 0 4px; font-size:14px; color:#fff; }
  #panel .check { margin-top:8px; }
  #panel .check input { margin-right:6px; }
  #hint { position:fixed; bottom:12px; left:12px; color:#666; font-size:12px; }
  #status { position:fixed; top:12px; left:12px; color:#0f0; font-size:13px; font-family:monospace; background:rgba(0,0,0,0.6); padding:6px 10px; border-radius:6px; }
</style>
<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
}}
</script>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="panel">
  <h3>PBR Parameters</h3>
  <label>Roughness <span class="val" id="r-val">0.50</span></label>
  <input type="range" id="roughness" min="0" max="1" step="0.01" value="0.5">
  <label>Metallic <span class="val" id="m-val">0.00</span></label>
  <input type="range" id="metallic" min="0" max="1" step="0.01" value="0.0">
  <label>Env Intensity <span class="val" id="e-val">1.00</span></label>
  <input type="range" id="envIntensity" min="0" max="3" step="0.05" value="1.0">
  <label>Exposure <span class="val" id="x-val">1.00</span></label>
  <input type="range" id="exposure" min="0.1" max="3" step="0.05" value="1.0">
  <label>Environment</label>
  <select id="env-select">__ENV_OPTIONS__</select>
  <div class="check"><label><input type="checkbox" id="skybox" checked> Show Skybox</label></div>
</div>
<div id="status">Loading...</div>
<div id="hint">Left-drag: rotate &middot; Right-drag: pan &middot; Scroll: zoom &middot; G: move &middot; R: rotate &middot; S: scale</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const ENV_PATHS = __ENV_PATHS__;
const DEFAULT_ENV = '__DEFAULT_ENV__';

const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.01, 100);
camera.position.set(0, 0.4, 1.5);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.25, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

const tControls = new TransformControls(camera, canvas);
tControls.setMode('translate');
tControls.addEventListener('dragging-changed', (e) => { controls.enabled = !e.value; });
scene.add(tControls);

const pmremGenerator = new THREE.PMREMGenerator(renderer);
pmremGenerator.compileEquirectangularShader();

let currentEnvTexture = null;
let model = null;
let origMats = [];
const status = document.getElementById('status');

function loadEnv(name) {
    const paths = ENV_PATHS[name];
    if (!paths) { status.textContent = 'ERROR: env not found: ' + name; return; }
    status.textContent = 'Loading env: ' + name + '...';
    const loader = new THREE.CubeTextureLoader();
    loader.load(
        [paths.px, paths.nx, paths.py, paths.ny, paths.pz, paths.nz],
        (cubeTexture) => {
            const envMap = pmremGenerator.fromCubemap(cubeTexture).texture;
            scene.environment = envMap;
            if (document.getElementById('skybox').checked) {
                scene.background = cubeTexture;
            }
            if (currentEnvTexture) currentEnvTexture.dispose();
            currentEnvTexture = envMap;
            scene.userData.cubeTexture = cubeTexture;
            status.textContent = 'Env: ' + name + ' OK';
        },
        undefined,
        (err) => { status.textContent = 'ENV FAILED: ' + err; status.style.color='#f44'; }
    );
}

loadEnv(DEFAULT_ENV);

const gltfLoader = new GLTFLoader();
status.textContent = 'Loading GLB...';
gltfLoader.load('__GLB_PATH__', (gltf) => {
    model = gltf.scene;
    scene.add(model);
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    controls.target.copy(center);
    camera.position.set(center.x, center.y + size.y * 0.3, center.z + Math.max(size.x, size.y, size.z) * 2);
    controls.update();
    let meshCount = 0;
    model.traverse((child) => {
        if (!child.isMesh) return;
        meshCount++;
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach(m => {
            if (m.isMeshStandardMaterial || m.isMeshPhysicalMaterial) {
                origMats.push({ mat: m, roughness: m.roughness, metalness: m.metalness, envMapIntensity: m.envMapIntensity });
            }
        });
    });
    status.textContent = 'OK: ' + meshCount + ' meshes, ' + origMats.length + ' PBR materials';
    origMats.forEach(o => console.log('Material:', o.mat.name, 'roughness:', o.roughness, 'metalness:', o.metalness));
    tControls.attach(model);
},
(progress) => {
    if (progress.total) status.textContent = 'GLB: ' + Math.round(progress.loaded/progress.total*100) + '%';
},
(err) => { status.textContent = 'GLB FAILED: ' + err; status.style.color='#f44'; }
);

function forEachOrigMat(fn) {
    origMats.forEach(o => fn(o));
}

document.getElementById('roughness').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    document.getElementById('r-val').textContent = v.toFixed(2);
    forEachOrigMat(o => { o.mat.roughness = v; });
});
document.getElementById('metallic').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    document.getElementById('m-val').textContent = v.toFixed(2);
    forEachOrigMat(o => { o.mat.metalness = v; });
});
document.getElementById('envIntensity').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    document.getElementById('e-val').textContent = v.toFixed(2);
    forEachOrigMat(o => { o.mat.envMapIntensity = v; });
});
document.getElementById('exposure').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    document.getElementById('x-val').textContent = v.toFixed(2);
    renderer.toneMappingExposure = v;
});
document.getElementById('env-select').addEventListener('change', (e) => {
    loadEnv(e.target.value);
});
document.getElementById('skybox').addEventListener('change', (e) => {
    if (e.target.checked && scene.userData.cubeTexture) {
        scene.background = scene.userData.cubeTexture;
    } else {
        scene.background = new THREE.Color(0x1a1a1a);
    }
});

window.addEventListener('keydown', (e) => {
    switch(e.key.toLowerCase()) {
        case 'g': tControls.setMode('translate'); break;
        case 'r': tControls.setMode('rotate'); break;
        case 's': tControls.setMode('scale'); break;
    }
});

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>"""

# 构建选项和替换
env_options = ''.join(f'<option value="{n}" {"selected" if n==default_env else ""}>{n}</option>' for n in env_list)
glb_path = "/assets/chair.glb"

viewer_page = VIEWER_HTML \
    .replace('__ENV_OPTIONS__', env_options) \
    .replace('__ENV_PATHS__', json.dumps(env_paths)) \
    .replace("'__DEFAULT_ENV__'", f"'{default_env}'") \
    .replace('__GLB_PATH__', glb_path)

viewer_path = '/tmp/chair_viewer.html'
with open(viewer_path, 'w') as f:
    f.write(viewer_page)

from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.routing import Route, Mount
import uvicorn

async def index(request):
    return FileResponse(viewer_path, media_type="text/html")

app = Starlette(routes=[
    Route("/", index),
    Mount("/assets", StaticFiles(directory=DESKTOP), name="assets"),
])

print("PBR Chair Viewer: http://localhost:7860")
uvicorn.run(app, host="0.0.0.0", port=7860)
