# LOCO//LAB

Real-time bipedal locomotion bench that runs **entirely in the browser**: MuJoCo 3.3
compiled to WebAssembly steps the physics of a 29-DoF Unitree G1, and a real neural
locomotion policy (ONNX, MLP 480→29, unitree_rl_gym-style velocity policy) closes the
loop at 50 Hz through onnxruntime-web. No server, no backend - open the page and shove a
humanoid.

Live: https://jacobegarcia.github.io/loco-lab/

## What you can do

- **Drag the robot** with the mouse to pull it off balance; watch the policy recover.
- **SHOVE** fires a random 150-250 N horizontal impulse at the torso (survivable - the policy was push-trained).
- **Speed / turn commands** feed the policy's velocity command input in real time.
- **Terrain** swaps the ground between flat, a 3° upgrade, and a 6° downgrade.
- Telemetry shows posture (UPRIGHT / STAGGER / DOWN), base height, commanded vs actual
  forward velocity, policy inference time, and realtime factor.

## Pipeline

- `policy.js` - observation construction (96-dim obs, frame stack of 5, group-major
  480-dim policy input), inference, position-target scaling, and joint-space PD at 500 Hz.
  Numerically validated against the reference MuJoCo deploy loop.
- `g1.zip` - G1 MJCF + 36 STL meshes + `policy.onnx`, unpacked into MuJoCo's MEMFS at boot.
- Rendering: three.js, with geometry read straight out of the compiled MuJoCo model.

The ONNX policy was converted from the public TorchScript checkpoint in
[RoboCubPilot/g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco)
(GPL-3.0), a MuJoCo deploy of the unitree_rl_gym G1 velocity policy. Sim-core approach
informed by [Axellwppr/humanoid-policy-viewer](https://github.com/Axellwppr/humanoid-policy-viewer)
and [ttktjmt/mjswan](https://github.com/ttktjmt/mjswan).
