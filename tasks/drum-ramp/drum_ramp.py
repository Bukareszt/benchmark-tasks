"""On the table: an orange drum lying on its side in front of the arms, a dark ramp rising away from the arms to a grey
deck, and a yellow stop bar along the far edge of the deck. The drum is too wide to grip and must be rolled.
Roll it up the ramp onto the deck until it rests against the stop, then move both arms away from it.
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
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# Layout, metres, world frame: x forward from the arms, y to the robot's left, z up.
DRUM_R = 0.10  # 20 cm across: far wider than the gripper's 94 mm opening
DRUM_LEN = 0.26
DRUM_MASS = 2.0
DRUM_X0 = 0.23  # starting centre
# The drum's surface friction with everything (static, dynamic). The rear of a rolling drum moves upwards, so a hand
# pushing it is dragged up by friction times the push; the arm's joints are soft enough that at the adapter's 1.2 the
# hand climbed the drum in pilot runs. A smooth plastic drum is modelled instead; see the fidelity note.
DRUM_FRICTION = (0.25, 0.20)
WIDTH = 0.50  # ramp, deck and stop, along y: 12 cm wider than the drum on each side, so the stop shows beside it from above
RAMP_X0 = 0.34  # foot of the ramp, where its top surface meets the table
RAMP_X1 = 0.45  # top of the ramp, where it meets the deck
DECK_H = 0.03
DECK_X1 = 0.66  # far edge of the deck
STOP_T = 0.02  # stop bar thickness along x; it stands on the deck's far edge
STOP_H = 0.03
RAMP_T = 0.012  # plank thickness
SLOPE = math.atan2(DECK_H, RAMP_X1 - RAMP_X0)

# Scoring thresholds.
ON_DECK_GAP = 0.005  # underside this close to the deck top counts as resting on it
ON_DECK_PAST_EDGE = 0.05  # the drum's centre at least this far past the ramp's top edge
ON_DECK_Y = 0.04  # and within this of the deck's centre line
AXIS_TOL_DEG = 15.0  # drum axis within this of the stop bar's direction (y)
CLEAR_M = 0.03  # no measured arm point (see _arm_state) this close to the drum: the arms have moved away
LIFT_M = 0.015  # centre this much higher than when rolling on the surface under it: lifted, 0 for the rest of the trial
PARTWAY_M = 0.015  # centre this much above its starting height at the end (half the deck height): 0.25
REST_SPEED = 0.01  # m/s
REST_SPIN = 0.5  # rad/s
# Keys this file's scene hook adds to the states passed to scoring.
TIPS = "tips"  # {"left": xyz, "right": xyz}: each gripper's fingertip midpoint, kept for the trace
ARM_POINTS = "arm_points"  # the arms' measured points, see _arm_state
ARM_POSES = "arm_poses"  # the link poses they were placed by, kept for the trace
TRIAL_STEP = "trial_step"  # the environment's step count since its last reset
STEP_DT = "step_dt"

INSTRUCTION = (
    "Roll the orange drum up the blue ramp onto the grey deck until it rests against the yellow stop bar, "
    "then move both arms at least 3 cm away from it."
)


@dataclass(frozen=True)
class Plank(Box):
    """A fixed box pitched about y: its +x end raised by ``pitch`` radians."""

    pitch: float = 0.0


@dataclass(frozen=True)
class Drum(Cylinder):
    """A cylinder lying along ``axis`` instead of standing upright, with its own surface friction (static, dynamic),
    which applies to everything it touches (PhysX combine mode "min")."""

    axis: str = "Y"
    friction: tuple[float, float] | None = None
    stripe: tuple[float, tuple[float, float, float]] | None = None  # (width, sRGB colour): a painted stripe along the drum
    linear_damping: float | None = None
    angular_damping: float | None = None


_link_vertices: dict[str, np.ndarray] = {}


def _arm_state(env) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Where the arms are after this step: every moving link's collision-mesh vertices, placed by the simulator's own
    link poses, and those poses. The fixed bases are left out. If either arm is missing a link there are no points,
    which clear() reads as not clear."""
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


def _striped_cylinder(scene, spec):
    """A cylinder spawner that paints a stripe along the body: a thin visual box riding on the surface, as the adapter
    paints its bottle's trim. It has no collider, so it changes nothing but the look."""
    width, color = spec.stripe

    @scene.clone
    def spawn(prim_path, cfg, *args, **kwargs):
        from pxr import Gf, UsdGeom

        prim = scene.spawn_cylinder.__wrapped__(prim_path, cfg, *args, **kwargs)
        stage = prim.GetStage()
        path = f"{prim_path}/trim/stripe"
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, spec.radius - 0.002))  # on top: the axis is y, so z is up
        geom.AddScaleOp().Set(Gf.Vec3f(width, spec.height * 0.98, 0.004))
        look = scene.sim_utils.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb(color), roughness=0.5)
        look.func(f"{path}_material", look)
        scene.sim_utils.bind_visual_material(path, f"{path}_material", stage=stage)
        return prim

    return spawn


def _extend_scene(scene) -> None:
    """Extend the adapter's scene module (revision 0059d02a1a91): its builder learns this file's two shapes, and the
    states it hands to scoring gain the fingertip midpoints, the trial's step count and the control period."""
    original = scene.task_object
    if getattr(original, "_drum_ramp", False):
        return

    def task_object(spec):
        if not isinstance(spec, Plank | Drum):
            return original(spec)
        look = scene.sim_utils.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb(spec.color), roughness=0.5)
        if isinstance(spec, Plank):
            cfg = scene.sim_utils.CuboidCfg(
                size=tuple(spec.size),
                collision_props=scene.PhysxCollisionPropertiesCfg(),
                physics_material=scene._grippy(),
                visual_material=look,
            )
            half = spec.pitch / 2
            rot = (0.0, -math.sin(half), 0.0, math.cos(half))
            kind = scene.ObjectType.BASE
        else:
            spawner = _striped_cylinder(scene, spec) if spec.stripe else scene.spawn_cylinder
            cfg = scene.sim_utils.CylinderCfg(
                func=spawner,
                radius=spec.radius,
                height=spec.height,
                axis=spec.axis,
                rigid_props=scene.PhysxRigidBodyPropertiesCfg(
                    linear_damping=spec.linear_damping, angular_damping=spec.angular_damping
                ),
                mass_props=scene.sim_utils.MassPropertiesCfg(mass=spec.mass),
                collision_props=scene._contact_props(),
                physics_material=scene._grippy()
                if spec.friction is None
                else scene.sim_utils.RigidBodyMaterialCfg(
                    static_friction=spec.friction[0], dynamic_friction=spec.friction[1], friction_combine_mode="min"
                ),
                visual_material=look,
            )
            rot = scene._yaw_xyzw(spec.yaw)
            kind = scene.ObjectType.RIGID
        return scene.Object(
            name=spec.name,
            object_type=kind,
            spawner_cfg=cfg,
            initial_pose=scene.Pose(position_xyz=tuple(spec.pos), rotation_xyzw=rot),
        )

    task_object._drum_ramp = True
    scene.task_object = task_object

    states = scene.object_states

    def object_states(env, names):
        from yam_arena.runtime import pad_tips_world

        base = env.unwrapped
        origin = base.scene.env_origins[0].detach().cpu().numpy()
        tips = {arm: pad_tips_world(base.scene[f"{arm}_robot"]) - origin for arm in ("left", "right")}
        points, poses = _arm_state(env)
        extra = {TIPS: tips, ARM_POINTS: points, ARM_POSES: poses, TRIAL_STEP: int(base.episode_length_buf[0]), STEP_DT: float(base.step_dt)}
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


_ramp_len = math.hypot(RAMP_X1 - RAMP_X0, DECK_H)
_ramp_mid = ((RAMP_X0 + RAMP_X1) / 2, 0.0, TOP + DECK_H / 2)  # centre of the plank's top surface
RAMP = Plank(
    "ramp",
    pos=(_ramp_mid[0] + RAMP_T / 2 * math.sin(SLOPE), 0.0, _ramp_mid[2] - RAMP_T / 2 * math.cos(SLOPE)),
    size=(_ramp_len, WIDTH, RAMP_T),
    color=(0.15, 0.35, 0.80),
    pitch=SLOPE,
)
DECK = Box("deck", pos=((RAMP_X1 + DECK_X1) / 2, 0.0, TOP + DECK_H / 2), size=(DECK_X1 - RAMP_X1, WIDTH, DECK_H), color=(0.62, 0.62, 0.60))
STOP = Box(
    "stop",
    pos=(DECK_X1 - STOP_T / 2, 0.0, TOP + DECK_H + STOP_H / 2),
    size=(STOP_T, WIDTH, STOP_H),
    color=(0.95, 0.80, 0.10),
)
DRUM = Drum(
    "drum",
    pos=(DRUM_X0, 0.0, TOP + DRUM_R + 0.001),
    radius=DRUM_R,
    height=DRUM_LEN,
    mass=DRUM_MASS,
    color=(0.90, 0.40, 0.08),
    friction=DRUM_FRICTION,
    stripe=(0.015, (0.20, 0.20, 0.22)),
)
OBJECTS = [RAMP, DECK, STOP, DRUM]
STOP_X0 = DECK_X1 - STOP_T  # the stop's face toward the ramp


def surface_z(x: float, y: float) -> float:
    """Height of the surface under a point: table, ramp or deck."""
    if abs(y) > WIDTH / 2 or x < RAMP_X0 or x > DECK_X1:
        return TOP
    if x < RAMP_X1:
        return TOP + (x - RAMP_X0) * math.tan(SLOPE)
    return TOP + DECK_H


def rolling_height(x: float, y: float) -> float:
    """Height of the drum's centre when it rolls on the surface under its centre."""
    if abs(y) <= WIDTH / 2 and RAMP_X0 <= x < RAMP_X1:
        return surface_z(x, y) + DRUM_R / math.cos(SLOPE)
    return surface_z(x, y) + DRUM_R


def axis_world(state) -> np.ndarray:
    """The drum's axis (its local y) in the world."""
    x, y, z, w = state.quat_xyzw
    return np.array([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)])


def axis_error_deg(state) -> float:
    a = axis_world(state)
    return math.degrees(math.acos(min(1.0, abs(float(a[1])) / float(np.linalg.norm(a)))))


def drum_distance(state, points) -> float:
    """Smallest distance from a point, or an (n, 3) array of points, to the drum's surface (a finite cylinder); 0 inside."""
    a = axis_world(state)
    a = a / np.linalg.norm(a)
    d = np.atleast_2d(np.asarray(points, float)) - state.pos
    along = d @ a
    radial = np.linalg.norm(d - np.outer(along, a), axis=1)
    dr = np.maximum(radial - DRUM_R, 0.0)
    da = np.maximum(np.abs(along) - DRUM_LEN / 2, 0.0)
    return float(np.hypot(dr, da).min())


def underside(state) -> float:
    """Height of the drum's lowest point: its centre less its radius. The adapter's bounding box is the drum's local
    box turned with it, which grows as the drum rolls, so it is not used for this."""
    return float(state.pos[2]) - DRUM_R


def on_deck(state) -> bool:
    """Resting on the deck top, its centre over the deck between the ramp's top edge and the stop, lying along the stop."""
    return (
        abs(underside(state) - (TOP + DECK_H)) <= ON_DECK_GAP
        and RAMP_X1 + ON_DECK_PAST_EDGE <= float(state.pos[0]) <= STOP_X0
        and abs(float(state.pos[1])) <= ON_DECK_Y
        and axis_error_deg(state) <= AXIS_TOL_DEG
    )


def arm_gap(objects) -> float:
    """Smallest distance from any measured arm point to the drum's surface; 0 when the arms were not measured."""
    points = objects.get(ARM_POINTS)
    if points is None:
        return 0.0
    return drum_distance(objects["drum"], points)


def clear(objects) -> bool:
    return arm_gap(objects) >= CLEAR_M


_memory = {"seen": None, "trial_step": None, "trial": 0, "lifted": False, "start_z": None, "steps": 0}


def _new_trial() -> None:
    _memory.update(lifted=False, start_z=None, trial_step=None)


def _update(objects) -> None:
    if _memory["seen"] is objects:
        return
    step = objects.get(TRIAL_STEP)
    if step is not None and _memory["trial_step"] is not None and step < _memory["trial_step"]:
        _new_trial()
        _memory["trial"] += 1
    _memory["seen"], _memory["trial_step"] = objects, step
    drum = objects["drum"]
    if _memory["start_z"] is None:
        _memory["start_z"] = float(drum.pos[2])
    if float(drum.pos[2]) > rolling_height(float(drum.pos[0]), float(drum.pos[1])) + LIFT_M:
        _memory["lifted"] = True


def success(objects) -> bool:
    """On the deck, still and let go; the adapter ends the episode once this has held for 1 s."""
    _update(objects)
    drum = objects["drum"]
    return not _memory["lifted"] and on_deck(drum) and drum.resting(REST_SPEED, REST_SPIN) and clear(objects)


def score(objects) -> float:
    _update(objects)
    drum = objects["drum"]
    if _memory["lifted"]:
        return 0.0
    if on_deck(drum):
        return 1.0 if drum.resting(REST_SPEED, REST_SPIN) and clear(objects) else 0.5
    if float(drum.pos[2]) - _memory["start_z"] >= PARTWAY_M:
        return 0.25
    return 0.0


def distance(objects) -> float:
    _log(objects)
    goal_x = (RAMP_X1 + DRUM_R + STOP_X0 - DRUM_R) / 2
    return float(abs(objects["drum"].pos[0] - goal_x))


def _log(objects) -> None:
    path = os.environ.get("DRUM_TRACE")
    if not path:
        return
    _update(objects)
    d = objects["drum"]
    tips = objects.get(TIPS) or {}
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if _memory["steps"] == 0:
            w.writerow(["step", "trial", "trial_step", "x", "y", "z", "qx", "qy", "qz", "qw", "speed", "spin", "low_z", "lifted",
                        "axis_deg", "on_deck", "lx", "ly", "lz", "rx", "ry", "rz", "gap_left", "gap_right", "arm_gap", "score"])
        l, r = tips.get("left", [np.nan] * 3), tips.get("right", [np.nan] * 3)
        w.writerow([_memory["steps"], _memory["trial"], objects.get(TRIAL_STEP), *np.round(d.pos, 6), *np.round(d.quat_xyzw, 5),
                    round(float(np.linalg.norm(d.lin_vel)), 5), round(float(np.linalg.norm(d.ang_vel)), 5), round(float(d.low[2]), 6),
                    int(_memory["lifted"]), round(axis_error_deg(d), 3), int(on_deck(d)), *np.round(l, 6), *np.round(r, 6),
                    round(drum_distance(d, l), 6) if tips else "", round(drum_distance(d, r), 6) if tips else "",
                    round(arm_gap(objects), 6), score(objects)])
    _log_arms(objects, Path(path))
    _memory["steps"] += 1


def _log_arms(objects, trace: Path) -> None:
    """Beside the trace: the link poses of this step, and once the hull vertices they place, so that arm_gap can be
    recomputed without this file. Pose translations are in the same frame as the trace; R is row-major."""
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

PUSH_Y = 0.09  # each arm pushes this far from the drum's middle, 4 cm in from its ends
# With the fingers closed and the grip turned so they close along x, the gripper body's face toward the drum is a
# flat vertical plate 5.1 cm from the TCP along the push, from 7 to 14 cm above the fingertips, with the finger
# carriage (4.8 cm from the TCP) just below it and the closed fingers hanging well behind. The fingertips are run
# along the surface under the hand (5 mm into it, so the arm rests on the surface), which puts the drum's axis 10 cm
# above the tips, against the plate, with 3 cm of plate below the axis; while the drum is on the ramp and the hand
# still on the table the axis sits up to 3 cm higher on the plate. A flat plate meets a cylinder at its axis height
# whatever the plate's height, so the push stays level. The closed fingers themselves widen with height, so they
# would meet a smaller drum above its axis and ride up over it, which pilot runs with a 10 cm drum showed; and
# pushing with the plate while the fingers hung in the air, the arm bent 2 to 4 cm upwards under the load and
# the plate slid up the drum.
FACE = 0.051  # the body face's distance from the TCP along the push
TIP_SINK = 0.01  # the fingertips are commanded this far into the surface under the hand
YAW = {"left": -math.pi / 2, "right": math.pi / 2}  # fingers closing along x; joint6 stays within its limits
PUSH_SPEED = 0.02  # m/s along the surface


def pusher(centre_x: float, arm: str, gap: float = 0.0) -> tuple[float, float, float]:
    """TCP position that puts the gripper body's face ``gap`` behind a drum whose centre is at ``centre_x``, with the
    fingertips on the surface under the hand."""
    y = PUSH_Y if arm == "left" else -PUSH_Y
    x = centre_x - DRUM_R - FACE - gap
    return (x, y, surface_z(x, y) + PAD_TIP_PAST_TCP - TIP_SINK)


def _raised(p, dz: float) -> tuple[float, float, float]:
    return (p[0], p[1], p[2] + dz)


def _travel(c0: float, c1: float) -> float:
    """Seconds to roll the drum's centre from c0 to c1 at PUSH_SPEED along the surface."""
    length = 0.0
    for a, b in ((c0, min(c1, RAMP_X0)), (max(c0, RAMP_X0), min(c1, RAMP_X1)), (max(c0, RAMP_X1), c1)):
        if b > a:
            length += (b - a) / (math.cos(SLOPE) if a >= RAMP_X0 and b <= RAMP_X1 else 1.0)
    return max(length / PUSH_SPEED, 0.5)


def demo():
    mode = os.environ.get("DEMO_MODE", "full")
    if mode in SHORTCUTS:
        return SHORTCUTS[mode]()
    end_c = float(os.environ.get("PUSH_TO", STOP_X0 - DRUM_R))  # where the drum's centre is pushed to
    plans = {}
    for arm in ("left", "right"):
        yaw = YAW[arm]
        start = pusher(DRUM_X0, arm, gap=0.02)
        plan = [
            Waypoint(2.0, (start[0], start[1], TOP + 0.18), FINGER_CLOSED, "above", yaw),
            Waypoint(1.5, start, FINGER_CLOSED, "down", yaw),
            Waypoint(1.0, pusher(DRUM_X0, arm), FINGER_CLOSED, "touch", yaw),
            Waypoint(_travel(DRUM_X0, RAMP_X0), pusher(RAMP_X0, arm), FINGER_CLOSED, "to the ramp", yaw),
            Waypoint(_travel(RAMP_X0, RAMP_X1), pusher(RAMP_X1, arm), FINGER_CLOSED, "up the ramp", yaw),
            Waypoint(_travel(RAMP_X1, end_c), pusher(end_c, arm), FINGER_CLOSED, "onto the deck", yaw),
            Waypoint(1.0, pusher(end_c, arm), FINGER_CLOSED, "settle", yaw),
            Waypoint(1.0, pusher(end_c, arm, gap=0.04), FINGER_CLOSED, "back off", yaw),
        ]
        if mode == "arm_stays":  # negative control: the fingers stay against the drum
            plan = plan[:-1]
        else:
            x, y, _ = pusher(end_c, arm, gap=0.04)
            plan += [
                Waypoint(1.0, (x, y, TOP + 0.13), FINGER_CLOSED, "up", yaw),
                Waypoint(1.5, (0.12, 0.22 if arm == "left" else -0.22, TOP + 0.18), FINGER_OPEN, "clear"),
            ]
        plans[arm] = plan
    if mode == "one_arm":  # negative control: the left arm alone pushes at the drum's middle
        return {"left": [Waypoint(w.duration, (w.pos[0], 0.0 if w.label != "clear" else 0.22, w.pos[2]), w.finger, w.label, w.yaw) for w in plans["left"]]}
    if mode == "one_end":  # negative control: the left arm alone pushes near the drum's left end
        return {"left": plans["left"]}
    return plans


# Shortcut attempts: motions the task does not intend, played as controls to see what they score. Each knob is an
# environment variable, so the receipts of a run say what it did.


def _knob(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _lift():
    """Both arms squeeze the drum's end faces with the gripper bodies (grip turned so the body's flat side faces the
    drum), raise it LIFT_M, carry it to the deck and set it down there. Tests the lifted rule."""
    squeeze = _knob("SQUEEZE_M", 0.03)  # commanded past the end face, since the joints give
    lift = _knob("LIFT_M", 0.05)
    end_c = _knob("PUSH_TO", RAMP_X1 + 0.03)  # as far as the arms reach while holding the drum by its ends
    side = 0.0325  # the body's half width across the fingers' closing direction
    z0 = TOP + DRUM_R - 0.07  # the body's flat side (7 to 14.5 cm above the TCP) centred on the axis
    z1 = TOP + DECK_H + DRUM_R - 0.07
    plans = {}
    for arm, sign in (("left", 1.0), ("right", -1.0)):
        y_out = sign * (DRUM_LEN / 2 + side + 0.03)
        y_in = sign * (DRUM_LEN / 2 + side - squeeze)
        yaw = YAW[arm]
        plans[arm] = [
            Waypoint(2.0, (DRUM_X0, y_out, TOP + 0.15), FINGER_CLOSED, "above", yaw),
            Waypoint(2.0, (DRUM_X0, y_out, z0), FINGER_CLOSED, "down", yaw),
            Waypoint(1.5, (DRUM_X0, y_in, z0), FINGER_CLOSED, "squeeze", yaw),
            Waypoint(2.0, (DRUM_X0, y_in, z0 + lift), FINGER_CLOSED, "lift", yaw),
            Waypoint(6.0, (end_c, y_in, z1 + lift), FINGER_CLOSED, "carry", yaw),
            Waypoint(2.0, (end_c, y_in, z1 + 0.005), FINGER_CLOSED, "lower", yaw),
            Waypoint(1.0, (end_c, y_out, z1 + 0.005), FINGER_CLOSED, "release", yaw),
            Waypoint(1.5, (end_c, y_out, TOP + 0.15), FINGER_CLOSED, "up", yaw),
            Waypoint(1.5, (0.12, sign * 0.22, TOP + 0.18), FINGER_OPEN, "clear"),
        ]
    return plans


def _knock():
    """Both arms strike the drum from behind at KNOCK_V m/s over KNOCK_M, then stop and withdraw: does momentum carry
    it up the ramp?"""
    v = _knob("KNOCK_V", 0.4)
    stroke = _knob("KNOCK_M", 0.10)
    plans = {}
    for arm in ("left", "right"):
        yaw = YAW[arm]
        start = pusher(DRUM_X0, arm, gap=0.03)
        hit = pusher(DRUM_X0 + stroke, arm)
        plans[arm] = [
            Waypoint(2.0, (start[0], start[1], TOP + 0.18), FINGER_CLOSED, "above", yaw),
            Waypoint(1.5, start, FINGER_CLOSED, "down", yaw),
            Waypoint((stroke + 0.03) / v, hit, FINGER_CLOSED, "strike", yaw),
            Waypoint(1.0, pusher(DRUM_X0 + stroke, arm, gap=0.05), FINGER_CLOSED, "back off", yaw),
            Waypoint(1.0, (hit[0] - 0.05, hit[1], TOP + 0.13), FINGER_CLOSED, "up", yaw),
            Waypoint(1.5, (0.12, 0.22 if arm == "left" else -0.22, TOP + 0.18), FINGER_OPEN, "clear"),
        ]
    return plans


SHORTCUTS = {"lift": _lift, "knock": _knock}
