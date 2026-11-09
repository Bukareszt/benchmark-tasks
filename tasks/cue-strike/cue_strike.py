"""On a green felt mat: a white ball near the arms, a square wooden stick lying to the right, a yellow line across the
mat, and beyond the line a red ball and a blue three-sided goal box with a light blue painted floor, open toward the arms,
inside low green rails.
Strike the white ball with the stick so that it knocks the red ball into the goal box. The stick and both arms stay
behind the yellow line, the arms never touch a ball, and the stick never touches the red ball.
"""

import csv
import importlib.abc
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box, Sphere

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# Layout, metres, world frame: x forward from the arms, y to the robot's left, z up.
MAT_T = 0.004  # felt mat thickness; everything below lies on it
MAT_X = (-0.06, 0.84)
MAT_Y = (-0.34, 0.34)
FLOOR = TOP + MAT_T  # the mat's top
BALL_R = 0.025
BALL_MASS = 0.16
WHITE0 = (0.30, -0.03)
RED0 = (0.53, 0.0)
LINE_X = 0.42  # the yellow line's centre; nothing of the stick or the arms may pass it
LINE_W = 0.015
GOAL_MOUTH_X = 0.70  # the goal box's open side, toward the arms
GOAL_DEPTH = 0.10  # inside, along x
GOAL_WIDTH = 0.14  # inside, along y
GOAL_Y = 0.14  # centre line of the goal box
WALL_T = 0.01
WALL_H = 0.03
STICK_W = 0.02  # square section
STICK_LEN = 0.34
STICK_MASS = 0.12
STICK0 = (0.15, -0.20)  # lying along x

# Ball physics: billiard-like balls bounce off each other; rolling resistance of the felt is approximated by damping.
BALL_FRICTION = (0.10, 0.08)
BALL_RESTITUTION = 0.9
BALL_DAMPING = (0.3, 0.3)  # linear, angular, 1/s

# Scoring thresholds.
ARM_TOUCH_M = 0.02  # no arm point may come this close to a ball's surface (mesh vertices miss mid-face contact)
STICK_TOUCH_M = 0.003  # the stick this close to the red ball touches it
REST_SPEED = 0.01
MOVED_M = 0.05  # red ball moved this far from its start: 0.25
NEAR_M = 0.10  # red ball's centre this close to the goal's inside centre: 0.5
ON_MAT_M = 0.005  # a ball's centre within this of resting on the mat

GOAL_X = (GOAL_MOUTH_X, GOAL_MOUTH_X + GOAL_DEPTH)  # inside, along x
GOAL_YS = (GOAL_Y - GOAL_WIDTH / 2, GOAL_Y + GOAL_WIDTH / 2)
GOAL_C = (GOAL_MOUTH_X + GOAL_DEPTH / 2, GOAL_Y)

# Keys this file's scene hook adds to the states passed to scoring.
ARM_POINTS = "arm_points"
ARM_POSES = "arm_poses"
TRIAL_STEP = "trial_step"
STEP_DT = "step_dt"

INSTRUCTION = (
    "Use the wooden stick to strike the white ball so that it knocks the red ball into the blue goal box, "
    "onto its light blue floor. Keep the stick and both arms on your side of the yellow line, keep both arms at "
    "least 2 cm from both balls, and never touch the red ball with the stick."
)


@dataclass(frozen=True)
class Ball(Sphere):
    """A billiard-like ball: low friction, bouncy, damped."""


@dataclass(frozen=True)
class Stick(Box):
    """The free wooden stick: a square rod lying along x."""


@dataclass(frozen=True)
class Mat(Box):
    """A fixed felt mat with a painted line across it (visual only, no collider)."""


@dataclass(frozen=True)
class Wall(Box):
    """A fixed wall that absorbs bounces (restitution 0, combine min)."""


_link_vertices: dict[str, np.ndarray] = {}


def _arm_state(env):
    """Every moving arm link's collision-mesh vertices placed by the simulator's link poses, and those poses."""
    from yam_arena.analysis import link_hulls
    from yam_arena.runtime import link_poses

    if not _link_vertices:
        for link, hulls in link_hulls().items():
            if link != "base":
                _link_vertices[link] = np.unique(np.concatenate([h.vertices for h in hulls]), axis=0)
    base = env.unwrapped
    origin = base.scene.env_origins[0].detach().cpu().numpy()
    points, poses = [], {}
    for arm in ("left_robot", "right_robot"):
        for link, pose in link_poses(base.scene[arm]).items():
            if link in _link_vertices:
                placed = pose.copy()
                placed[:3, 3] -= origin
                poses[f"{arm}/{link}"] = placed
                points.append(_link_vertices[link] @ placed[:3, :3].T + placed[:3, 3])
    if len(poses) != 2 * len(_link_vertices):
        return None, poses
    return np.concatenate(points), poses


def _mat_spawner(scene, spec):
    @scene.clone
    def spawn(prim_path, cfg, *args, **kwargs):
        from isaaclab.sim.spawners.shapes import spawn_cuboid
        from pxr import Gf, UsdGeom

        prim = spawn_cuboid.__wrapped__(prim_path, cfg, *args, **kwargs)
        stage = prim.GetStage()
        path = f"{prim_path}/marks/line"
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
        cx = (MAT_X[0] + MAT_X[1]) / 2
        geom.AddTranslateOp().Set(Gf.Vec3d(LINE_X - cx, 0.0, MAT_T / 2 + 0.0004))
        geom.AddScaleOp().Set(Gf.Vec3f(LINE_W, (MAT_Y[1] - MAT_Y[0]) * 0.999, 0.0006))
        look = scene.sim_utils.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb((0.95, 0.82, 0.10)), roughness=0.6)
        look.func(f"{path}_material", look)
        scene.sim_utils.bind_visual_material(path, f"{path}_material", stage=stage)
        # the goal box's floor, painted so that "inside" can be seen
        path = f"{prim_path}/marks/goal_floor"
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(GOAL_C[0] - cx, GOAL_C[1], MAT_T / 2 + 0.0004))
        geom.AddScaleOp().Set(Gf.Vec3f(GOAL_DEPTH, GOAL_WIDTH, 0.0006))
        look = scene.sim_utils.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb((0.62, 0.80, 0.95)), roughness=0.6)
        look.func(f"{path}_material", look)
        scene.sim_utils.bind_visual_material(path, f"{path}_material", stage=stage)
        return prim

    return spawn


def _extend_scene(scene) -> None:
    """Extend the adapter's scene module (revision 0059d02a1a91): the builder learns this file's shapes, and the
    states passed to scoring gain the arms' measured points, the trial's step count and the control period."""
    original = scene.task_object
    if getattr(original, "_cue_strike", False):
        return
    su = scene.sim_utils

    def task_object(spec):
        if not isinstance(spec, Ball | Stick | Mat | Wall):
            return original(spec)
        look = su.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb(spec.color), roughness=0.4)
        rot = scene._yaw_xyzw(spec.yaw)
        if isinstance(spec, Ball):
            cfg = su.SphereCfg(
                radius=spec.radius,
                rigid_props=scene.PhysxRigidBodyPropertiesCfg(
                    linear_damping=BALL_DAMPING[0], angular_damping=BALL_DAMPING[1], sleep_threshold=0.0, stabilization_threshold=0.0
                ),
                mass_props=su.MassPropertiesCfg(mass=spec.mass),
                collision_props=scene._contact_props(),
                physics_material=su.RigidBodyMaterialCfg(
                    static_friction=BALL_FRICTION[0], dynamic_friction=BALL_FRICTION[1], restitution=BALL_RESTITUTION
                ),
                visual_material=look,
            )
            kind = scene.ObjectType.RIGID
        elif isinstance(spec, Stick):
            cfg = su.CuboidCfg(
                size=tuple(spec.size),
                rigid_props=scene.PhysxRigidBodyPropertiesCfg(),
                mass_props=su.MassPropertiesCfg(mass=spec.mass),
                collision_props=scene._contact_props(),
                physics_material=scene._grippy(),
                visual_material=look,
            )
            kind = scene.ObjectType.RIGID
        else:
            material = su.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0, restitution=0.0, restitution_combine_mode="min"
            )
            cfg = su.CuboidCfg(
                size=tuple(spec.size),
                collision_props=scene.PhysxCollisionPropertiesCfg(),
                physics_material=material,
                visual_material=look,
            )
            if isinstance(spec, Mat):
                cfg = su.CuboidCfg(
                    func=_mat_spawner(scene, spec),
                    size=tuple(spec.size),
                    collision_props=scene.PhysxCollisionPropertiesCfg(),
                    physics_material=material,
                    visual_material=look,
                )
            kind = scene.ObjectType.BASE
        return scene.Object(
            name=spec.name,
            object_type=kind,
            spawner_cfg=cfg,
            initial_pose=scene.Pose(position_xyz=tuple(spec.pos), rotation_xyzw=rot),
        )

    task_object._cue_strike = True
    scene.task_object = task_object
    states = scene.object_states

    def object_states(env, names):
        points, poses = _arm_state(env)
        base = env.unwrapped
        extra = {ARM_POINTS: points, ARM_POSES: poses, TRIAL_STEP: int(base.episode_length_buf[0]), STEP_DT: float(base.step_dt)}
        return {**states(env, names), **extra}

    scene.object_states = object_states


class _SceneHook(importlib.abc.MetaPathFinder):
    """yam_arena.scene needs a running simulator, so it is extended as it is imported, not when this file loads."""

    def find_spec(self, name, path=None, target=None):
        if name != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _extend_scene(module)

        spec.loader.exec_module = exec_module
        return spec


if "yam_arena.scene" in sys.modules:
    _extend_scene(sys.modules["yam_arena.scene"])
elif not any(isinstance(f, _SceneHook) for f in sys.meta_path):
    sys.meta_path.insert(0, _SceneHook())


MAT = Mat(
    "mat",
    pos=((MAT_X[0] + MAT_X[1]) / 2, (MAT_Y[0] + MAT_Y[1]) / 2, TOP + MAT_T / 2),
    size=(MAT_X[1] - MAT_X[0], MAT_Y[1] - MAT_Y[0], MAT_T),
    color=(0.10, 0.45, 0.22),
)
_wz = FLOOR + WALL_H / 2
_blue = (0.15, 0.35, 0.85)
GOAL_WALLS = [
    Wall("goal_back", pos=(GOAL_X[1] + WALL_T / 2, GOAL_Y, _wz), size=(WALL_T, GOAL_WIDTH + 2 * WALL_T, WALL_H), color=_blue),
    Wall("goal_left", pos=((GOAL_X[0] + GOAL_X[1]) / 2, GOAL_YS[1] + WALL_T / 2, _wz), size=(GOAL_DEPTH, WALL_T, WALL_H), color=_blue),
    Wall("goal_right", pos=((GOAL_X[0] + GOAL_X[1]) / 2, GOAL_YS[0] - WALL_T / 2, _wz), size=(GOAL_DEPTH, WALL_T, WALL_H), color=_blue),
]
WHITE = Ball("white", pos=(*WHITE0, FLOOR + BALL_R + 0.0005), radius=BALL_R, mass=BALL_MASS, color=(0.93, 0.93, 0.90))
RED = Ball("red", pos=(*RED0, FLOOR + BALL_R + 0.0005), radius=BALL_R, mass=BALL_MASS, color=(0.80, 0.08, 0.08))
STICK = Stick("stick", pos=(*STICK0, FLOOR + STICK_W / 2 + 0.0005), size=(STICK_LEN, STICK_W, STICK_W), mass=STICK_MASS,
              color=(0.40, 0.22, 0.10))
RAIL_H = 0.02
_rz = FLOOR + RAIL_H / 2
_rail = (0.08, 0.30, 0.15)
_side_x0 = LINE_X + 0.05
RAILS = [
    Wall("rail_far", pos=(MAT_X[1] - WALL_T / 2, 0.0, _rz), size=(WALL_T, MAT_Y[1] - MAT_Y[0], RAIL_H), color=_rail),
    Wall("rail_left", pos=((_side_x0 + MAT_X[1]) / 2, MAT_Y[1] - WALL_T / 2, _rz), size=(MAT_X[1] - _side_x0, WALL_T, RAIL_H), color=_rail),
    Wall("rail_right", pos=((_side_x0 + MAT_X[1]) / 2, MAT_Y[0] + WALL_T / 2, _rz), size=(MAT_X[1] - _side_x0, WALL_T, RAIL_H), color=_rail),
]
OBJECTS = [MAT, *GOAL_WALLS, *RAILS, WHITE, RED, STICK]


# -- scoring ----------------------------------------------------------------------------------------------------


def _rot(state) -> np.ndarray:
    x, y, z, w = np.asarray(state.quat_xyzw, float) / np.linalg.norm(state.quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


STICK_HALF = np.array([STICK_LEN / 2, STICK_W / 2, STICK_W / 2])
_CORNERS = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)


def stick_front_x(state) -> float:
    """The largest x of any point of the stick: its foremost corner."""
    return float(((_CORNERS * STICK_HALF) @ _rot(state).T + state.pos)[:, 0].max())


def stick_ball_gap(stick, ball) -> float:
    """Distance from the stick's surface (a box) to a ball's surface; 0 or less when they touch."""
    local = (np.asarray(ball.pos, float) - stick.pos) @ _rot(stick)
    return float(np.linalg.norm(np.maximum(np.abs(local) - STICK_HALF, 0.0))) - BALL_R


def arm_ball_gap(objects, name) -> float:
    points = objects.get(ARM_POINTS)
    if points is None:
        return 0.0
    return float(np.linalg.norm(points - objects[name].pos, axis=1).min() - BALL_R)


def arm_front_x(objects) -> float:
    points = objects.get(ARM_POINTS)
    return float("inf") if points is None else float(points[:, 0].max())


def in_goal(ball) -> bool:
    x, y, z = (float(v) for v in ball.pos)
    return (
        GOAL_X[0] + BALL_R <= x <= GOAL_X[1]
        and GOAL_YS[0] <= y <= GOAL_YS[1]
        and abs(z - (FLOOR + BALL_R)) <= ON_MAT_M
    )


def goal_distance(ball) -> float:
    return float(math.hypot(ball.pos[0] - GOAL_C[0], ball.pos[1] - GOAL_C[1]))


_memory = {"seen": None, "trial_step": None, "trial": 0, "foul": None, "red0": None, "steps": 0}


def _new_trial() -> None:
    _memory.update(foul=None, red0=None, trial_step=None)


def fouls(objects) -> list[str]:
    out = []
    if stick_front_x(objects["stick"]) > LINE_X - LINE_W / 2:
        out.append("stick_over_line")
    if arm_front_x(objects) > LINE_X - LINE_W / 2:
        out.append("arm_over_line")
    for name in ("white", "red"):
        if arm_ball_gap(objects, name) < ARM_TOUCH_M:
            out.append(f"arm_touched_{name}")
    if stick_ball_gap(objects["stick"], objects["red"]) < STICK_TOUCH_M:
        out.append("stick_touched_red")
    return out


def _update(objects) -> None:
    if _memory["seen"] is objects:
        return
    step = objects.get(TRIAL_STEP)
    if step is not None and _memory["trial_step"] is not None and step < _memory["trial_step"]:
        _new_trial()
        _memory["trial"] += 1
    _memory["seen"], _memory["trial_step"] = objects, step
    if _memory["red0"] is None:
        _memory["red0"] = np.array(objects["red"].pos, float)
    found = fouls(objects)
    if found and _memory["foul"] is None:
        _memory["foul"] = ",".join(found)


def success(objects) -> bool:
    """Red ball still inside the goal box and no foul so far; the adapter ends the episode once this has held 1 s."""
    _update(objects)
    red = objects["red"]
    return _memory["foul"] is None and in_goal(red) and red.resting(REST_SPEED, 0.5)


def score(objects) -> float:
    _update(objects)
    red = objects["red"]
    if _memory["foul"] is not None:
        return 0.0
    if in_goal(red) and red.resting(REST_SPEED, 0.5):
        return 1.0
    if goal_distance(red) <= NEAR_M:
        return 0.5
    if np.linalg.norm(np.asarray(red.pos[:2]) - _memory["red0"][:2]) >= MOVED_M:
        return 0.25
    return 0.0


def distance(objects) -> float:
    _log(objects)
    return goal_distance(objects["red"])


def _log(objects) -> None:
    path = os.environ.get("CUE_TRACE")
    if not path:
        return
    _update(objects)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if _memory["steps"] == 0:
            w.writerow(["step", "trial", "trial_step",
                        *[f"{o}_{k}" for o in ("white", "red", "stick") for k in ("x", "y", "z", "qx", "qy", "qz", "qw", "speed", "spin")],
                        "stick_front_x", "arm_front_x", "arm_gap_white", "arm_gap_red", "stick_gap_red", "foul", "in_goal", "score"])
        row = [_memory["steps"], _memory["trial"], objects.get(TRIAL_STEP)]
        for name in ("white", "red", "stick"):
            o = objects[name]
            row += [*np.round(o.pos, 6), *np.round(o.quat_xyzw, 6), round(float(np.linalg.norm(o.lin_vel)), 6),
                    round(float(np.linalg.norm(o.ang_vel)), 6)]
        row += [round(stick_front_x(objects["stick"]), 6), round(arm_front_x(objects), 6), round(arm_ball_gap(objects, "white"), 6),
                round(arm_ball_gap(objects, "red"), 6), round(stick_ball_gap(objects["stick"], objects["red"]), 6),
                _memory["foul"] or "", int(in_goal(objects["red"])), score(objects)]
        w.writerow(row)
    _log_arms(objects, Path(path))
    _memory["steps"] += 1


def _log_arms(objects, trace: Path) -> None:
    poses = objects.get(ARM_POSES)
    if not poses:
        return
    first = _memory["steps"] == 0
    if first:
        with open(trace.with_suffix(".hulls.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["link", "vx", "vy", "vz"])
            for link, vertices in _link_vertices.items():
                w.writerows([link, *v] for v in vertices.tolist())
    with open(trace.with_suffix(".arms.csv"), "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["step", "arm", "link", "x", "y", "z", *[f"r{i}{j}" for i in range(3) for j in range(3)]])
        for name, pose in poses.items():
            w.writerow([_memory["steps"], *name.split("/"), *np.round(pose[:3, 3], 6), *np.round(pose[:3, :3].ravel(), 6)])


# -- demo -------------------------------------------------------------------------------------------------------


def _knob(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def aim():
    """Where the white ball must go: the ghost-ball point, one diameter behind the red ball on the red-to-goal line;
    returns (unit direction white -> ghost, distance)."""
    red = np.array(RED0)
    to_goal = np.array(GOAL_C) - red
    to_goal /= np.linalg.norm(to_goal)
    ghost = red - 2 * BALL_R * to_goal
    d = ghost - np.array(WHITE0)
    return d / np.linalg.norm(d), float(np.linalg.norm(d))


def demo():
    """The right arm picks the stick up near its back, turns it along the aim line, lowers its tip to the white
    ball's centre height and strikes; then it backs off and puts the stick down. The left arm stays where it is."""
    mode = os.environ.get("DEMO_MODE", "full")
    if mode in SHORTCUTS:
        return SHORTCUTS[mode]()
    if mode == "idle":
        return {"right": [Waypoint(3.0, (0.25, -0.25, TOP + 0.16), FINGER_OPEN, "idle")]}
    return _stick_plan(aim_off=mode == "aim_off")


def _stick_plan(aim_off=False, aim_deg=None, stroke=None, strike_s=None):
    grip_back = _knob("GRIP_BACK", 0.09)  # grip this far behind the stick's centre
    stroke = _knob("STROKE_M", 0.10) if stroke is None else stroke  # TCP travel during the strike
    strike_s = _knob("STRIKE_S", 0.25) if strike_s is None else strike_s
    gap = _knob("GAP_M", 0.04)  # tip-to-ball gap before the strike
    dirn, _ = aim()
    if aim_off:  # a negative control: aim straight at the red ball's centre
        d = np.array(RED0) - np.array(WHITE0)
        dirn = d / np.linalg.norm(d)
    # Aim correction found by trial: the held stick turns in the grip as it strikes.
    turn = math.radians(_knob("AIM_DEG", 2.2) if aim_deg is None else aim_deg)
    dirn = np.array([dirn[0] * math.cos(turn) - dirn[1] * math.sin(turn), dirn[0] * math.sin(turn) + dirn[1] * math.cos(turn)])
    yaw = math.atan2(dirn[1], dirn[0])
    tip_from_tcp = STICK_LEN / 2 + grip_back
    pick = (STICK0[0] - grip_back, STICK0[1], FLOOR + PAD_TIP_PAST_TCP + 0.004)
    hit_dz = _knob("HIT_DZ", 0.009)  # TCP above the stick's axis while held
    high = TOP + 0.16
    white = np.array(WHITE0)
    contact = white - dirn * BALL_R  # where the tip meets the ball
    hit_z = FLOOR + BALL_R + hit_dz  # stick axis at the ball's centre height
    ready = contact - dirn * (gap + tip_from_tcp)
    through = ready + dirn * stroke
    right = [
        Waypoint(2.0, (pick[0], pick[1], high), FINGER_OPEN, "above stick"),
        Waypoint(1.5, pick, FINGER_OPEN, "down"),
        Waypoint(0.8, pick, FINGER_CLOSED, "close"),
        Waypoint(1.2, (pick[0], pick[1], FLOOR + 0.08), FINGER_CLOSED, "lift"),
        Waypoint(2.0, (ready[0], ready[1], FLOOR + 0.08), FINGER_CLOSED, "to aim", yaw),
        Waypoint(1.2, (ready[0], ready[1], hit_z), FINGER_CLOSED, "lower", yaw),
        Waypoint(0.6, (ready[0], ready[1], hit_z), FINGER_CLOSED, "settle", yaw),
        Waypoint(strike_s, (through[0], through[1], hit_z), FINGER_CLOSED, "strike", yaw),
        Waypoint(0.8, (ready[0], ready[1], hit_z), FINGER_CLOSED, "back", yaw),
        Waypoint(1.0, (ready[0], ready[1], FLOOR + PAD_TIP_PAST_TCP + 0.006), FINGER_CLOSED, "put down", yaw),
        Waypoint(0.6, (ready[0], ready[1], FLOOR + PAD_TIP_PAST_TCP + 0.006), FINGER_OPEN, "release", yaw),
        Waypoint(1.2, (ready[0], ready[1], high), FINGER_OPEN, "up", yaw),
    ]
    return {"right": right}


# Shortcut attempts: motions the task does not intend, played to see what they score. Knobs are environment variables.


def _shot_line():
    dirn, _ = aim()
    return np.array(WHITE0), dirn


def _hand_push():
    """Strike the white ball with the closed fingers instead of the stick, along the aim line, and withdraw."""
    white, dirn = _shot_line()
    z = FLOOR + BALL_R + PAD_TIP_PAST_TCP  # fingertips at the ball's centre height
    a = white - dirn * (BALL_R + 0.04)
    b = white + dirn * _knob("STROKE_M", 0.04)
    yaw = math.atan2(dirn[1], dirn[0]) + math.pi / 2  # fingers closing across the line, pads facing the ball
    return {"right": [
        Waypoint(2.0, (a[0], a[1], TOP + 0.16), FINGER_CLOSED, "above", yaw),
        Waypoint(1.5, (a[0], a[1], z), FINGER_CLOSED, "down", yaw),
        Waypoint(_knob("STRIKE_S", 0.2), (b[0], b[1], z), FINGER_CLOSED, "strike", yaw),
        Waypoint(1.0, (a[0], a[1], z), FINGER_CLOSED, "back", yaw),
        Waypoint(1.5, (a[0], a[1], TOP + 0.16), FINGER_OPEN, "up", yaw),
    ]}


def _stick_push():
    """The full demo's pick-up, then push the white ball with the stick's tip slowly all the way to the ghost point,
    which takes the stick over the line."""
    return _stick_plan(aim_deg=0.0, stroke=_knob("PUSH_M", aim()[1] + 0.04), strike_s=_knob("PUSH_S", 6.0))


def _reach_red():
    """The left arm reaches over the line and pushes the red ball into the goal with its closed fingers."""
    red = np.array(RED0)
    to_goal = np.array(GOAL_C) - red
    to_goal /= np.linalg.norm(to_goal)
    z = FLOOR + BALL_R + PAD_TIP_PAST_TCP + _knob("REACH_DZ", 0.0)
    a = red - to_goal * (BALL_R + 0.04)
    b = np.array(GOAL_C) - to_goal * BALL_R
    yaw = math.atan2(to_goal[1], to_goal[0]) + math.pi / 2
    arm = os.environ.get("REACH_ARM", "left")
    return {arm: [
        Waypoint(2.0, (a[0], a[1], TOP + 0.16), FINGER_CLOSED, "above", yaw),
        Waypoint(1.5, (a[0], a[1], z), FINGER_CLOSED, "down", yaw),
        Waypoint(4.0, (b[0], b[1], z), FINGER_CLOSED, "push", yaw),
        Waypoint(1.5, (b[0] - 0.1, b[1], TOP + 0.16), FINGER_OPEN, "up", yaw),
    ]}


SHORTCUTS = {"hand_push": _hand_push, "stick_push": _stick_push, "reach_red": _reach_red}
