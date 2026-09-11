// LOCO//LAB - velocity-policy controller for the Unitree G1 (29 dof).
// Mirrors the validated deploy pipeline (unitree_rl_gym-style) exactly:
// 96-dim obs, frame stack of 5, group-major 480-dim policy input,
// position targets from scaled policy output, joint-space PD at 500 Hz.

export const CFG = {
  timestep: 0.002,
  decimation: 10,           // 50 Hz policy
  numActions: 29,
  numObs: 96,
  stack: 5,
  angVelScale: 0.2,
  dofVelScale: 0.05,
  actionScale: 0.25,
  kps: [100,100,100,150,40,40,100,100,100,150,40,40,200,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40],
  kds: [2,2,2,4,2,2,2,2,2,4,2,2,5,5,5,10,10,10,10,10,10,10,10,10,10,10,10,10,10],
  defaultAnglesPolicyOrder: [-0.1,-0.1,0,0,0,0,0,0,0,0.3,0.3,0.3,0.3,-0.2,-0.2,0.25,-0.25,0,0,0,0,0.97,0.97,0.15,-0.15,0,0,0,0],
  policyToXml: [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28],
  xmlToPolicy: [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28],
};

const NA = CFG.numActions;

// default angles reordered to mujoco (xml) order
export const DEFAULT_MJC = CFG.policyToXml.map((pi) => CFG.defaultAnglesPolicyOrder[pi]);

function gravityOrientation(qw, qx, qy, qz) {
  return [
    2 * (-qz * qx + qw * qy),
    -2 * (qz * qy + qw * qx),
    1 - 2 * (qw * qw + qz * qz),
  ];
}

export class VelocityPolicy {
  constructor(session, ortLib) {
    this.session = session;
    this.ort = ortLib;      // onnxruntime InferenceSession (input 'obs' [1,480])
    this.frames = [];            // last 5 obs vectors (Float32Array(96)), oldest first
    this.action = new Float32Array(NA);   // last raw policy output, policy order
    this.target = Float32Array.from(DEFAULT_MJC);  // position targets, mujoco order
    this.input = new Float32Array(1 * 480);
    this.inferMs = 0;
    this.reset();
  }

  reset() {
    this.frames = [];
    for (let i = 0; i < CFG.stack; i++) this.frames.push(new Float32Array(CFG.numObs));
    this.action.fill(0);
    this.target.set(DEFAULT_MJC);
  }

  // Build the 96-dim obs from raw mujoco state.
  buildObs(qpos, qvel, cmd) {
    const obs = new Float32Array(CFG.numObs);
    const g = gravityOrientation(qpos[3], qpos[4], qpos[5], qpos[6]);
    for (let i = 0; i < 3; i++) {
      obs[i] = qvel[3 + i] * CFG.angVelScale;
      obs[3 + i] = g[i];
      obs[6 + i] = cmd[i];
    }
    for (let i = 0; i < NA; i++) {
      const x = CFG.xmlToPolicy[i];
      obs[9 + i] = (qpos[7 + x] - DEFAULT_MJC[x]);            // dof pos, policy order
      obs[9 + NA + i] = qvel[6 + x] * CFG.dofVelScale;        // dof vel, policy order
      obs[9 + 2 * NA + i] = this.action[i];                   // last action, policy order (validated pipeline)
    }
    return obs;
  }

  // One 50 Hz control tick. Returns position targets in mujoco order.
  async tick(qpos, qvel, cmd) {
    this.frames.push(this.buildObs(qpos, qvel, cmd));
    if (this.frames.length > CFG.stack) this.frames.shift();

    // group-major stack: [omega x5, grav x5, cmd x5, pos x5, vel x5, act x5]
    const big = this.input;
    let o = 0;
    const groups = [[0, 3], [3, 6], [6, 9], [9, 9 + NA], [9 + NA, 9 + 2 * NA], [9 + 2 * NA, 9 + 3 * NA]];
    for (const [a, b] of groups) {
      for (const f of this.frames) {
        for (let i = a; i < b; i++) big[o++] = f[i];
      }
    }

    const t0 = performance.now();
    const out = await this.session.run({ obs: new this.ort.Tensor('float32', big, [1, 480]) });
    this.inferMs = performance.now() - t0;
    const a = out.action.data;
    for (let i = 0; i < NA; i++) this.action[i] = a[i];

    for (let i = 0; i < NA; i++) {
      this.target[i] = a[CFG.policyToXml[i]] * CFG.actionScale + DEFAULT_MJC[i];
    }
    return this.target;
  }

  // Joint-space PD toward current targets; writes ctrl (mujoco order).
  pd(qpos, qvel, ctrl) {
    for (let i = 0; i < NA; i++) {
      ctrl[i] = CFG.kps[i] * (this.target[i] - qpos[7 + i]) - CFG.kds[i] * qvel[6 + i];
    }
  }
}
