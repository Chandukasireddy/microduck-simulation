"""Franka Panda arm picks up a cube, carries it, drops it, and repeats.

Physics: MuJoCo simulates gravity, joints, motors, and the cube's contact with the gripper.
Arm motion: a scripted controller (damped least-squares inverse kinematics on the
tool-center point). No AI training is involved.

Run:  python demo_arm_pick.py
The viewer stays open and keeps repeating. Close the window to stop.
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).parent / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"

CUBE_START = np.array([0.5, 0.0, 0.02])  # cube center, resting on the floor
CUBE_HALF_SIZE = 0.02
CARRY_TO = np.array([0.5, 0.25, 0.02])  # where the cube is dropped
HOME_JOINTS = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])

# Gripper actuator (actuator8) range is 0 to 255. From the Panda model comments:
# 255 = fingers open, 0 = fingers closed.
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

# (phase name, tool-center target in meters, gripper command, steps of 2 ms each)
PHASES = [
    ("move above cube", CUBE_START + [0, 0, 0.15], GRIPPER_OPEN, 1500),
    ("lower to cube", CUBE_START, GRIPPER_OPEN, 1000),
    ("close gripper", CUBE_START, GRIPPER_CLOSED, 600),
    ("lift", CUBE_START + [0, 0, 0.2], GRIPPER_CLOSED, 1000),
    ("carry", CARRY_TO + [0, 0, 0.2], GRIPPER_CLOSED, 1200),
    ("lower", CARRY_TO, GRIPPER_CLOSED, 1000),
    ("release", CARRY_TO, GRIPPER_OPEN, 600),
    ("retreat", CARRY_TO + [0, 0, 0.2], GRIPPER_OPEN, 800),
]


def build_model() -> mujoco.MjModel:
    """Load the Panda scene and add a cube and a tool-center-point marker."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.body("hand").add_site(
        name="tcp", pos=[0, 0, 0.1034], size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 1]
    )
    cube = spec.worldbody.add_body(name="cube", pos=CUBE_START.tolist())
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF_SIZE] * 3,
        rgba=[1, 0.4, 0.1, 1],
        mass=0.05,
    )
    return spec.compile()


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the arm in its home pose, open the gripper, and place the cube on the floor."""
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME_JOINTS
    data.qpos[7:9] = 0.04  # finger joints: open
    data.ctrl[:7] = HOME_JOINTS
    data.ctrl[7] = GRIPPER_OPEN
    data.joint("cube_joint").qpos[:3] = CUBE_START
    mujoco.mj_forward(model, data)


DOWN = np.array([0.0, 0.0, -1.0])  # the gripper points straight down while grasping


def arm_command(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, target: np.ndarray) -> np.ndarray:
    """Joint targets that move the tool center to the target and keep the gripper pointing down."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp, jacr])[:, :7]  # position and orientation rows for the seven arm joints

    pos_error = target - data.site_xpos[site_id]
    approach_axis = data.site_xmat[site_id].reshape(3, 3)[:, 2]  # the gripper's pointing direction
    rot_error = np.cross(approach_axis, DOWN)  # rotation needed to point it down
    error = np.concatenate([2.0 * pos_error, 1.0 * rot_error])

    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), error)
    dq = np.clip(dq, -0.05, 0.05)  # limit how fast the joints can move
    return data.qpos[:7] + dq


def run(show: bool = True, cycles: int | None = None) -> dict:
    """Run the pick-and-place loop. Returns simple stats for checking the result."""
    model = build_model()
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    stats = {"max_cube_height_while_carrying": 0.0, "cube_final_position": None}

    viewer = mujoco.viewer.launch_passive(model, data) if show else None
    cycle = 0
    try:
        while cycles is None or cycle < cycles:
            reset(model, data)
            for name, target, gripper, steps in PHASES:
                print(f"Cycle {cycle + 1}: {name}")
                for _ in range(steps):
                    if viewer is not None and not viewer.is_running():
                        return stats
                    started = time.perf_counter()

                    data.ctrl[:7] = arm_command(model, data, site_id, target)
                    data.ctrl[7] = gripper
                    mujoco.mj_step(model, data)

                    if name in ("lift", "carry"):
                        height = data.joint("cube_joint").qpos[2]
                        stats["max_cube_height_while_carrying"] = max(
                            stats["max_cube_height_while_carrying"], height
                        )
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - started)))
            cycle += 1
    finally:
        if viewer is not None:
            viewer.close()
        stats["cube_final_position"] = data.joint("cube_joint").qpos[:3].copy()
    return stats


if __name__ == "__main__":
    run(show=True)
