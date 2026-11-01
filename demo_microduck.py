"""MicroDuck: a 25 cm bipedal robot simulated in MuJoCo.

The robot model comes from Pollen Robotics' open-source microduck_rl repo
(assets/microduck_rl). The robot starts in its FOLD pose, which holds still
without a trained policy. The demo moves the head side to side so you can
see the joints and physics working.

Note: this demo does NOT make the duck walk. Walking needs a trained policy
(see the microduck_rl repo). It only shows the robot model and its motors.

Run:  python demo_microduck.py
The viewer stays open. Close the window to stop.
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = (
    Path(__file__).parent
    / "assets"
    / "microduck_rl"
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "microduck"
    / "scene_walk.xml"
)

HEAD_YAW_INDEX = 7  # actuator index for head_yaw (joint order is fixed in the model)
HEAD_YAW_SWING = 0.6  # radians the head turns left and right
SWING_PERIOD_S = 4.0


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE))


def reset_to_fold(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Put the robot in its FOLD pose and return that pose's motor commands."""
    key = model.key("FOLD")
    mujoco.mj_resetDataKeyframe(model, data, key.id)
    data.ctrl[:] = key.ctrl
    return key.ctrl.copy()


def run(show: bool = True, seconds: float | None = None) -> dict:
    """Show the robot. Returns the final trunk height and upright check for testing."""
    model = load_model()
    data = mujoco.MjData(model)
    fold_ctrl = reset_to_fold(model, data)
    trunk = model.body("trunk_base").id

    viewer = mujoco.viewer.launch_passive(model, data) if show else None
    try:
        steps = int(seconds / model.opt.timestep) if seconds is not None else None
        step = 0
        while steps is None or step < steps:
            if viewer is not None and not viewer.is_running():
                break
            started = time.perf_counter()

            t = data.time
            ctrl = fold_ctrl.copy()
            ctrl[HEAD_YAW_INDEX] += HEAD_YAW_SWING * np.sin(2 * np.pi * t / SWING_PERIOD_S)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            step += 1

            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - started)))
    finally:
        if viewer is not None:
            viewer.close()

    up_z = data.xmat[trunk].reshape(3, 3)[2, 2]  # 1.0 means the robot is upright
    return {"trunk_height_m": float(data.qpos[2]), "upright": bool(up_z > 0.9)}


if __name__ == "__main__":
    run(show=True)
