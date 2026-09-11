// LOCO//LAB - real-time MuJoCo WASM + ONNX locomotion bench for the Unitree G1.
import * as THREE from './three.module.js';
import { OrbitControls } from './OrbitControls.js';
import loadMujoco from 'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js';
import * as ort from './ort.min.mjs';
import { CFG, DEFAULT_MJC, VelocityPolicy } from './policy.js';

// ---------- scene XML variants ----------
const SCENE_HEAD = `<mujoco model="g1_loco">
  <include file="g1_29dof.xml"/>
  <option timestep="0.002"/>
  <statistic center="0 0 0.5" extent="2.0"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
`;
const SCENE_TAIL = `
  </worldbody>
</mujoco>
`;
const TERRAINS = {
  flat: {
    label: 'FLAT',
    geom: '<geom name="floor" size="0 0 0.05" type="plane"/>',
  },
  uphill: {
    label: 'UPHILL 3\u00B0',
    geom: '<body name="grade" pos="0 0 0" euler="0 0.0524 0"><geom name="floor" size="40 40 0.05" type="box" pos="0 0 -0.0501"/></body>',
  },
  downhill: {
    label: 'DOWNHILL 6\u00B0',
    geom: '<body name="grade" pos="0 0 0" euler="0 -0.1047 0"><geom name="floor" size="40 40 0.05" type="box" pos="0 0 -0.0501"/></body>',
  },
};
function sceneXML(key) {
  return SCENE_HEAD + '    ' + TERRAINS[key].geom + '\n' + SCENE_TAIL;
}

// ---------- boot log ----------
const bootEl = document.getElementById('bootlog');
const bootBox = document.getElementById('boot');
function blog(msg) {
  const line = document.createElement('div');
  line.textContent = '> ' + msg;
  bootEl.appendChild(line);
  bootEl.scrollTop = bootEl.scrollHeight;
}
function setStatus(txt, cls) {
  const pill = document.getElementById('status');
  pill.textContent = txt;
  pill.className = 'pill ' + cls;
}

// ---------- telemetry ----------
const T = {};
for (const id of ['rt', 'infer', 'basez', 'velx', 'velxbar', 'yaw', 'simt', 'fallstat']) {
  T[id] = document.getElementById('t-' + id);
}

// ---------- main ----------
const state = {
  paused: false,
  cmd: [0.5, 0, 0],
  terrain: 'flat',
  follow: true,
  shoveUntil: -1,
  shoveVec: [0, 0],
  fallSince: null,
  fallen: false,
  alive: false,
  simTime: 0,
  rt: 1,
};

let mujoco, model, data, policy;
let bodies = {}, mujocoRoot = null;
let torsoBodyId = -1, pelvisBodyId = 1;

async function boot() {
  try {
    blog('loading mujoco wasm runtime (11 MB)');
    mujoco = await loadMujoco();
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
    blog('wasm runtime online');

    blog('fetching g1 assets (mjcf + 36 meshes + onnx policy)');
    const ASSETS = ["g1_29dof.xml", "policy.onnx", "meshes/head_link.STL", "meshes/left_ankle_pitch_link.STL", "meshes/left_ankle_roll_link.STL", "meshes/left_elbow_link.STL", "meshes/left_hip_pitch_link.STL", "meshes/left_hip_roll_link.STL", "meshes/left_hip_yaw_link.STL", "meshes/left_knee_link.STL", "meshes/left_rubber_hand.STL", "meshes/left_shoulder_pitch_link.STL", "meshes/left_shoulder_roll_link.STL", "meshes/left_shoulder_yaw_link.STL", "meshes/left_wrist_pitch_link.STL", "meshes/left_wrist_roll_link.STL", "meshes/left_wrist_yaw_link.STL", "meshes/logo_link.STL", "meshes/pelvis.STL", "meshes/pelvis_contour_link.STL", "meshes/right_ankle_pitch_link.STL", "meshes/right_ankle_roll_link.STL", "meshes/right_elbow_link.STL", "meshes/right_hip_pitch_link.STL", "meshes/right_hip_roll_link.STL", "meshes/right_hip_yaw_link.STL", "meshes/right_knee_link.STL", "meshes/right_rubber_hand.STL", "meshes/right_shoulder_pitch_link.STL", "meshes/right_shoulder_roll_link.STL", "meshes/right_shoulder_yaw_link.STL", "meshes/right_wrist_pitch_link.STL", "meshes/right_wrist_roll_link.STL", "meshes/right_wrist_yaw_link.STL", "meshes/torso_link.STL", "meshes/waist_roll_link.STL", "meshes/waist_support_link.STL", "meshes/waist_yaw_link.STL"];
    if (!mujoco.FS.analyzePath('/working/meshes').exists) mujoco.FS.mkdir('/working/meshes');
    const bufs = await Promise.all(ASSETS.map(async (p) => {
      const r = await fetch('./assets/' + p);
      if (!r.ok) throw new Error('asset fetch failed: ' + p + ' (http ' + r.status + ')');
      return [p, new Uint8Array(await r.arrayBuffer())];
    }));
    let onnxBuf = null, meshCount = 0;
    for (const [p, buf] of bufs) {
      if (p === 'policy.onnx') { onnxBuf = buf; continue; }
      mujoco.FS.writeFile('/working/' + p, buf);
      if (p.endsWith('.STL')) meshCount++;
    }
    blog('assets in memfs: ' + meshCount + ' meshes + robot MJCF + policy.onnx');

    blog('loading onnx policy (MLP 480 -> 29, unitree rl gym velocity)');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.1/dist/';
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(onnxBuf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    policy = new VelocityPolicy(session, ort);
    blog('onnx session ready [' + session.inputNames[0] + ' -> ' + session.outputNames[0] + ']');

    initGraphics();
    await loadTerrain('flat');

    blog('first policy tick...');
    await policyTick();
    blog('online. drag the robot to perturb it.');
    bootBox.classList.add('hidden');
    setStatus('RUNNING', 'ok');
    state.alive = true;
    mainLoop();
  } catch (err) {
    blog('BOOT FAILURE: ' + (err && err.message ? err.message : err));
    setStatus('BOOT ERROR', 'bad');
    console.error(err);
  }
}

// ---------- three.js ----------
let scene, camera, renderer, controls, drag;
const container = document.getElementById('view');

function initGraphics() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070a);
  scene.fog = new THREE.Fog(0x05070a, 18, 46);

  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 200);
  camera.position.set(2.6, 1.7, 2.6);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.8, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.maxPolarAngle = Math.PI * 0.52;
  controls.update();

  const hemi = new THREE.HemisphereLight(0x8fa3bf, 0x14110c, 0.55);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xfff2dd, 1.35);
  key.position.set(4, 7, 3);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -6; key.shadow.camera.right = 6;
  key.shadow.camera.top = 6; key.shadow.camera.bottom = -6;
  key.shadow.camera.far = 30;
  key.shadow.bias = -0.0004;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x6f87ff, 0.35);
  rim.position.set(-5, 3, -4);
  scene.add(rim);

  const grid = new THREE.GridHelper(80, 80, 0x1c2431, 0x10151d);
  grid.position.y = 0.001;
  scene.add(grid);

  drag = new DragPerturb(scene, renderer, camera, container, controls);

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
  renderer.setAnimationLoop(renderFrame);
}

function buildRobotFromModel() {
  if (mujocoRoot) {
    scene.remove(mujocoRoot);
    mujocoRoot.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
  }
  bodies = {};
  mujocoRoot = new THREE.Group();
  mujocoRoot.name = 'MuJoCo Root';
  scene.add(mujocoRoot);

  const textDecoder = new TextDecoder();
  const namesArray = new Uint8Array(model.names);
  const meshes = {};

  for (let g = 0; g < model.ngeom; g++) {
    if (!(model.geom_group[g] < 3)) continue;
    const b = model.geom_bodyid[g];
    const type = model.geom_type[g];
    const size = [model.geom_size[g * 3], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]];

    if (!(b in bodies)) {
      bodies[b] = new THREE.Group();
      let s = model.name_bodyadr[b], e = s;
      while (e < namesArray.length && namesArray[e] !== 0) e++;
      bodies[b].name = textDecoder.decode(namesArray.subarray(s, e));
      bodies[b].bodyID = b;
      if (bodies[b].name === 'pelvis') pelvisBodyId = b;
      if (bodies[b].name === 'torso_link') torsoBodyId = b;
    }

    let geometry = null;
    if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
      geometry = new THREE.PlaneGeometry(90, 90);
    } else if (type === mujoco.mjtGeom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
    } else if (type === mujoco.mjtGeom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(size[0], 20, 20);
    } else if (type === mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 16);
    } else if (type === mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 20);
    } else if (type === mujoco.mjtGeom.mjGEOM_MESH.value) {
      const meshID = model.geom_dataid[g];
      if (!(meshID in meshes)) {
        geometry = new THREE.BufferGeometry();
        const vbuf = model.mesh_vert.slice(
          model.mesh_vertadr[meshID] * 3,
          (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3);
        const nbuf = model.mesh_normal.slice(
          model.mesh_vertadr[meshID] * 3,
          (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3);
        for (let v = 0; v < vbuf.length; v += 3) {
          let t = vbuf[v + 1]; vbuf[v + 1] = vbuf[v + 2]; vbuf[v + 2] = -t;
          t = nbuf[v + 1]; nbuf[v + 1] = nbuf[v + 2]; nbuf[v + 2] = -t;
        }
        const fbuf = model.mesh_face.slice(
          model.mesh_faceadr[meshID] * 3,
          (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(vbuf, 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(nbuf, 3));
        geometry.setIndex(Array.from(fbuf));
        meshes[meshID] = geometry;
      } else {
        geometry = meshes[meshID];
      }
    }
    if (!geometry) continue;

    const rgba = [
      model.geom_rgba[g * 4], model.geom_rgba[g * 4 + 1],
      model.geom_rgba[g * 4 + 2], model.geom_rgba[g * 4 + 3]];
    let material;
    if (type === mujoco.mjtGeom.mjGEOM_PLANE.value || (b === 0 || (b in bodies && bodies[b].name === 'grade'))) {
      material = new THREE.MeshStandardMaterial({ color: 0x0b0e13, roughness: 0.95, metalness: 0.0 });
    } else {
      const isTorso = bodies[b].name === 'torso_link';
      material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]).multiplyScalar(0.9),
        roughness: 0.42,
        metalness: 0.55,
      });
      if (isTorso) material.color.setHex(0x2a2f38);
    }
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = type !== mujoco.mjtGeom.mjGEOM_PLANE.value;
    mesh.receiveShadow = true;
    mesh.bodyID = b;
    if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
      mesh.rotateX(-Math.PI / 2);
    } else {
      setGeomPos(model.geom_pos, g, mesh.position);
      setGeomQuat(model.geom_quat, g, mesh.quaternion);
    }
    if (type === mujoco.mjtGeom.mjGEOM_BOX.value && b === 0) {
      // static box under plane - not used
    }
    bodies[b].add(mesh);
  }

  for (let b = 0; b < model.nbody; b++) {
    if (b === 0 || !bodies[0]) mujocoRoot.add(bodies[b] || new THREE.Group());
    else if (bodies[b]) bodies[0].add(bodies[b]);
  }
}

function setGeomPos(buf, i, target) { target.set(buf[i * 3], buf[i * 3 + 2], -buf[i * 3 + 1]); }
function setGeomQuat(buf, i, target) { target.set(-buf[i * 4 + 1], -buf[i * 4 + 3], buf[i * 4 + 2], -buf[i * 4]); }
function bodyPos(buf, i, target) { target.set(buf[i * 3], buf[i * 3 + 2], -buf[i * 3 + 1]); }
function bodyQuat(buf, i, target) { target.set(-buf[i * 4 + 1], -buf[i * 4 + 3], buf[i * 4 + 2], -buf[i * 4]); }

// ---------- terrain / reset ----------
async function loadTerrain(key) {
  state.terrain = key;
  if (data) { data.delete(); model.delete(); data = null; model = null; }
  mujoco.FS.writeFile('/working/scene.xml', sceneXML(key));
  model = mujoco.MjModel.loadFromXML('/working/scene.xml');
  data = new mujoco.MjData(model);
  torsoBodyId = -1; pelvisBodyId = 1;
  buildRobotFromModel();
  resetSim();
}

function resetSim() {
  mujoco.mj_resetData(model, data);
  for (let i = 0; i < CFG.numActions; i++) data.qpos[7 + i] = DEFAULT_MJC[i];
  mujoco.mj_forward(model, data);
  policy.reset();
  state.fallen = false;
  state.fallSince = null;
  state.simTime = 0;
  setStatus(state.paused ? 'PAUSED' : 'RUNNING', state.paused ? 'warn' : 'ok');
}

// ---------- drag perturbation ----------
class DragPerturb {
  constructor(scene, renderer, camera, container, controls) {
    this.scene = scene; this.renderer = renderer; this.camera = camera; this.controls = controls;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.active = false; this.obj = null; this.grabDist = 0;
    this.localHit = new THREE.Vector3(); this.worldHit = new THREE.Vector3(); this.cur = new THREE.Vector3();
    this.arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xffb000, 0.12, 0.06);
    this.arrow.visible = false;
    scene.add(this.arrow);
    const el = renderer.domElement;
    el.addEventListener('pointerdown', (e) => this.down(e));
    document.addEventListener('pointermove', (e) => this.move(e));
    document.addEventListener('pointerup', () => this.up());
  }
  ray(x, y) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.mouse, this.camera);
  }
  down(e) {
    this.ray(e.clientX, e.clientY);
    const hits = this.raycaster.intersectObjects(this.scene.children, true);
    for (const h of hits) {
      if (h.object.bodyID && h.object.bodyID > 0) {
        this.obj = h.object; this.grabDist = h.distance; this.active = true;
        this.controls.enabled = false;
        this.localHit.copy(h.object.worldToLocal(h.point.clone()));
        this.worldHit.copy(h.point); this.cur.copy(h.point);
        this.arrow.visible = true;
        break;
      }
    }
  }
  move(e) {
    if (!this.active) return;
    this.ray(e.clientX, e.clientY);
    this.cur.copy(this.raycaster.ray.origin).addScaledVector(this.raycaster.ray.direction, this.grabDist);
  }
  up() { this.active = false; this.obj = null; this.controls.enabled = true; this.arrow.visible = false; }
  // returns mujoco-frame force + point, or null
  force() {
    if (!this.active || !this.obj) return null;
    this.worldHit.copy(this.localHit);
    this.obj.localToWorld(this.worldHit);
    const off = this.cur.clone().sub(this.worldHit);
    this.arrow.position.copy(this.worldHit);
    if (off.lengthSq() > 1e-6) this.arrow.setDirection(off.clone().normalize());
    this.arrow.setLength(Math.min(off.length(), 1.2) + 0.001, 0.1, 0.05);
    // three (y-up) -> mujoco (z-up): [x, -z, y]
    let f = [off.x * 160, -off.z * 160, off.y * 160];
    const mag = Math.hypot(...f);
    const MAX = 180;
    if (mag > MAX) f = f.map((v) => (v * MAX) / mag);
    return { f, point: [this.worldHit.x, -this.worldHit.z, this.worldHit.y], body: this.obj.bodyID };
  }
}

// ---------- sim loop ----------
const _forceBuf = new Float64Array(3);
const _torqueBuf = new Float64Array(3);
const _pointBuf = new Float64Array(3);

async function policyTick() {
  await policy.tick(data.qpos, data.qvel, state.cmd);
}

async function mainLoop() {
  const dec = CFG.decimation;
  while (state.alive) {
    const t0 = performance.now();
    if (!state.paused && model && data) {
      await policyTick();

      for (let s = 0; s < dec; s++) {
        policy.pd(data.qpos, data.qvel, data.ctrl);

        const qa = data.qfrc_applied;
        for (let i = 0; i < qa.length; i++) qa[i] = 0;
        const xa = data.xfrc_applied;
        for (let i = 0; i < xa.length; i++) xa[i] = 0;

        const df = drag.force();
        if (df) {
          _forceBuf[0] = df.f[0]; _forceBuf[1] = df.f[1]; _forceBuf[2] = df.f[2];
          _torqueBuf[0] = 0; _torqueBuf[1] = 0; _torqueBuf[2] = 0;
          _pointBuf[0] = df.point[0]; _pointBuf[1] = df.point[1]; _pointBuf[2] = df.point[2];
          mujoco.mj_applyFT(model, data, _forceBuf, _torqueBuf, _pointBuf, df.body, data.qfrc_applied);
        }
        if (state.simTime < state.shoveUntil && torsoBodyId >= 0) {
          xa[torsoBodyId * 6] = state.shoveVec[0];
          xa[torsoBodyId * 6 + 1] = state.shoveVec[1];
        }

        mujoco.mj_step(model, data);
        state.simTime += CFG.timestep;
      }

      // fall detection
      const z = data.qpos[2];
      if (z < 0.4) {
        if (state.fallSince === null) state.fallSince = state.simTime;
        if (!state.fallen && state.simTime - state.fallSince > 0.5) {
          state.fallen = true;
          setStatus('FALLEN', 'bad');
        }
      } else {
        state.fallSince = null;
        if (state.fallen) { state.fallen = false; setStatus('RUNNING', 'ok'); }
      }
      updateTelemetry(t0);
    }
    const elapsed = performance.now() - t0;
    const target = CFG.timestep * dec * 1000;
    await new Promise((r) => setTimeout(r, Math.max(0, target - elapsed)));
  }
}

const _pv = new THREE.Vector3();
function renderFrame() {
  if (!model || !data) return;
  for (let b = 0; b < model.nbody; b++) {
    if (!bodies[b]) continue;
    bodyPos(data.xpos, b, bodies[b].position);
    bodyQuat(data.xquat, b, bodies[b].quaternion);
    bodies[b].updateWorldMatrix();
  }
  if (state.follow && bodies[pelvisBodyId]) {
    _pv.set(bodies[pelvisBodyId].position.x, 0.8, bodies[pelvisBodyId].position.z);
    const dx = _pv.x - controls.target.x, dz = _pv.z - controls.target.z;
    controls.target.x += dx * 0.08; controls.target.z += dz * 0.08;
    camera.position.x += dx * 0.08; camera.position.z += dz * 0.08;
  }
  controls.update();
  renderer.render(scene, camera);
}

let lastTel = 0, wallPrev = performance.now(), simPrev = 0;
function updateTelemetry(t0) {
  if (t0 - lastTel < 200) return;
  const now = performance.now();
  state.rt = ((state.simTime - simPrev) / ((now - wallPrev) / 1000));
  wallPrev = now; simPrev = state.simTime; lastTel = t0;

  T.rt.textContent = state.rt.toFixed(2) + 'x';
  T.infer.textContent = policy.inferMs.toFixed(1) + ' ms';
  T.basez.textContent = data.qpos[2].toFixed(3) + ' m';
  const vx = data.qvel[0];
  T.velx.textContent = state.cmd[0].toFixed(2) + ' / ' + vx.toFixed(2) + ' m/s';
  const pct = Math.max(0, Math.min(1, state.cmd[0] ? vx / state.cmd[0] : 0));
  T.velxbar.style.width = (pct * 100).toFixed(0) + '%';
  T.yaw.textContent = state.cmd[2].toFixed(2) + ' rad/s';
  T.simt.textContent = state.simTime.toFixed(1) + ' s';
  T.fallstat.textContent = state.fallen ? 'DOWN' : (state.fallSince !== null ? 'STAGGER' : 'UPRIGHT');
  T.fallstat.className = 'val ' + (state.fallen ? 'bad' : (state.fallSince !== null ? 'warn' : ''));
}

// ---------- UI wiring ----------
function bind(id, ev, fn) { document.getElementById(id).addEventListener(ev, fn); }

bind('btn-play', 'click', () => {
  state.paused = !state.paused;
  document.getElementById('btn-play').textContent = state.paused ? 'RESUME' : 'PAUSE';
  setStatus(state.paused ? 'PAUSED' : 'RUNNING', state.paused ? 'warn' : 'ok');
});
bind('btn-reset', 'click', () => resetSim());
bind('btn-shove', 'click', () => {
  const ang = Math.random() * Math.PI * 2;
  const mag = 150 + Math.random() * 100;
  state.shoveVec = [Math.cos(ang) * mag, Math.sin(ang) * mag];
  state.shoveUntil = state.simTime + 0.1;
  flashShove(mag);
});
bind('spd', 'input', (e) => {
  state.cmd[0] = parseFloat(e.target.value);
  document.getElementById('spd-val').textContent = state.cmd[0].toFixed(2) + ' m/s';
});
bind('yaw', 'input', (e) => {
  state.cmd[2] = parseFloat(e.target.value);
  document.getElementById('yaw-val').textContent = state.cmd[2].toFixed(2) + ' rad/s';
});
bind('follow', 'change', (e) => { state.follow = e.target.checked; });
for (const key of Object.keys(TERRAINS)) {
  document.getElementById('ter-' + key).addEventListener('click', async (e) => {
    if (state.terrain === key) return;
    for (const k of Object.keys(TERRAINS)) document.getElementById('ter-' + k).classList.remove('active');
    e.target.classList.add('active');
    setStatus('LOADING', 'warn');
    await loadTerrain(key);
    setStatus('RUNNING', 'ok');
  });
}
function flashShove(mag) {
  const el = document.getElementById('shove-flash');
  el.textContent = mag.toFixed(0) + 'N';
  el.classList.remove('show');
  void el.offsetWidth;
  el.classList.add('show');
}

boot();
