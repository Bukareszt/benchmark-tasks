"""On the table: a blue block resting on three loose orange wooden rollers that lie across its path, and two green
strips marking a zone further along the table, on the robot's right, which is the right of the top camera's picture.
The block is too wide to grip with one hand.
Roll it into the zone without letting it touch the table; a roller freed behind the block has to be moved to
the front before the block runs off the leading one. Finish with both arms let go and moved clear.
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

ROLLER_R = 0.015
ROLLER_LEN = 0.26  # sticks out 7 cm past each block side; the finger body is +-34 mm wide in x above the tip
ROLLER_MASS = 0.08
BLOCK_SIZE = (0.12, 0.16, 0.08)  # x, y, z, m; wider than the gripper's 94 mm opening both ways seen from above
BLOCK_MASS = 3.0
LANE_X = 0.25
START_Y = 0.07
SPACING = 0.065
BLOCK_Z = TOP + 2 * ROLLER_R + BLOCK_SIZE[2] / 2 + 0.001

ZONE_Y = -0.14
ZONE_LEN = 0.22
STRIP_X = 0.145  # strip centres, either side of the lane and clear of the roller ends
STRIP_WIDTH = 0.02
ZONE_X_HALF = STRIP_X - STRIP_WIDTH / 2  # to the strips' inner edges

# Scoring thresholds, all measurable with a tape measure and photos or video frames.
TIER_1_M = 0.06  # block travel worth 0.25
TIER_2_M = 0.15  # worth 0.50: past the 0.137 m measured before a block runs off its leading roller
ON_TABLE_M = 0.020  # block underside this close to the table: it came off the rollers
LIFTED_M = 0.035  # underside this high: it was raised, not rolled; the demo stays at 2.97-3.00 cm, on 3 cm rollers
SLIP_MAX_M = 0.02  # block travel not explained by rolling, over the trial; see _slip_step(). The demo's is about 2.5 mm
MAX_BLOCK_SPEED = 0.2  # m/s, centre speed: above this it was struck, not rolled. The demo's fastest is 4.2 cm/s
MAX_TILT_DEG = 5.0
ROLLERS_NEEDED = 2
UNDER_Z_SLACK = 0.005  # a roller more than this above lying on the table is not under the block, it is being carried
REST_SPEED = 0.01  # m/s, the fastest any point of the block may move; see block_speed()
CLEAR_M = 0.05  # no measured arm point (see _arm_state) this close to the block or any roller: it has moved clear
CONTROL_HZ = 20  # the rate this task is run at
HOLD_S = 1.0  # the end conditions hold this long, from the first step to the last
HOLD_STEPS = round(HOLD_S * CONTROL_HZ) + 1  # consecutive control steps at CONTROL_HZ
SUCCESS_HOLD_S = 0.0  # read by the adapter: success() below does the holding, so the adapter ends the episode at once
# Keys of what this file's scene hook adds to the states passed to scoring.
ARM_POINTS = "arm_points"  # the arms' measured points
ARM_POSES = "arm_poses"  # the link poses they were placed by, kept for the trace
TRIAL_STEP = "trial_step"  # the environment's step count since its last reset
STEP_DT = "step_dt"  # the environment's own control period, s

INSTRUCTION = (
    "In the top camera view, roll the blue block to the right until all of it lies between the two green strips and "
    "within their length, at rest on at least two of the three orange rollers. The rollers are loose: carry a roller "
    "from behind the block to the front of it whenever the block needs one. The block must never touch the table. At "
    "the end, let go and move both arms at least 5 cm away from the block and the rollers."
)

TRAVEL_Z = TOP + 0.16
FIRST_PUSH = 0.08
TOTAL_PUSH = 0.215
GRIP_X = LANE_X - 0.105
GRIP_OPEN = -0.027  # finger joint target: 27 mm each side of the 30 mm roller, instead of the full 47 mm
PICK_Y = START_Y + SPACING - 0.0284  # measured: where the rear roller stops once the block has left it


@dataclass(frozen=True)
class Block(Box):
    linear_damping: float | None = None  # 1/s; stands in for rolling resistance


@dataclass(frozen=True)
class Roller(Cylinder):
    """A capsule along ``axis``: PhysX collides it analytically, so unlike a cylinder it does not creep on its side."""

    axis: str = "X"
    linear_damping: float | None = None
    angular_damping: float | None = None


_link_vertices: dict[str, np.ndarray] = {}


def _arm_state(env) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Where the arms are after this step: every moving link's collision-mesh vertices, placed by the simulator's own
    link poses, and those poses. The fixed bases are left out, since they cannot be moved clear of anything. If either
    arm is missing a link there are no points, which arm_gap reads as not clear."""
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


def _extend_scene(scene) -> None:
    """Extend the adapter's scene module (revision 0059d02a1a91), with its own helpers: its builder learns this file's
    two shapes, and the states it hands to scoring gain the arms' measured points, the trial's step count and the
    control period."""
    original = scene.task_object
    if getattr(original, "_roller_block", False):
        return

    def task_object(spec):
        if not isinstance(spec, Block | Roller):
            return original(spec)
        body = dict(
            rigid_props=scene.PhysxRigidBodyPropertiesCfg(
                linear_damping=spec.linear_damping, angular_damping=getattr(spec, "angular_damping", None)
            ),
            mass_props=scene.sim_utils.MassPropertiesCfg(mass=spec.mass),
            collision_props=scene._contact_props(),
            physics_material=scene._grippy(),
            visual_material=scene.sim_utils.PreviewSurfaceCfg(diffuse_color=scene.linear_rgb(spec.color), roughness=0.4),
        )
        if isinstance(spec, Block):
            cfg = scene.sim_utils.CuboidCfg(size=tuple(spec.size), **body)
        else:
            cfg = scene.sim_utils.CapsuleCfg(radius=spec.radius, height=spec.height, axis=spec.axis, **body)
        return scene.Object(
            name=spec.name,
            object_type=scene.ObjectType.RIGID,
            spawner_cfg=cfg,
            initial_pose=scene.Pose(position_xyz=tuple(spec.pos), rotation_xyzw=scene._yaw_xyzw(spec.yaw)),
        )

    task_object._roller_block = True
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


ROLLERS = [
    Roller(f"roller_{i}", pos=(LANE_X, START_Y + dy, TOP + ROLLER_R + 0.0005), radius=ROLLER_R, height=ROLLER_LEN - 2 * ROLLER_R,
           mass=ROLLER_MASS, color=(0.90, 0.55, 0.10), linear_damping=0.5, angular_damping=3.0)
    for i, dy in enumerate((-SPACING, 0.0, SPACING))
]
BLOCK = Block("block", pos=(LANE_X, START_Y, BLOCK_Z), size=BLOCK_SIZE, mass=BLOCK_MASS, color=(0.12, 0.25, 0.75), linear_damping=4.0)
# Strips outside the rollers' lane: a plate under them would tilt the block, and the rollers' viscous damping cannot
# hold it on a slope, however slight.
ZONE = [
    Box(f"zone_{side}", pos=(LANE_X + sign * STRIP_X, ZONE_Y, TOP + 0.0005), size=(STRIP_WIDTH, ZONE_LEN, 0.001), color=(0.10, 0.70, 0.20))
    for side, sign in (("near", -1), ("far", 1))
]
OBJECTS = [*ZONE, *ROLLERS, BLOCK]
ROLLER_NAMES = tuple(r.name for r in ROLLERS)

_memory = {"failed": False, "steps": 0, "trial": 0, "held": 0, "gap": 0.0, "complete": False, "done": False, "seen": None, "trial_step": None,
           "start_y": None, "last_y": None, "slip": 0.0}


def _new_trial() -> None:
    """Forget the last trial. Only the environment's own reset does this: putting the block back does not undo a failure."""
    _memory.update(failed=False, held=0, gap=0.0, complete=False, done=False, seen=None, trial_step=None, start_y=None, last_y=None, slip=0.0)


def tilt_deg(state) -> float:
    """Angle between the block's up axis and the vertical."""
    x, y, _, _ = state.quat_xyzw
    return math.degrees(math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _underside(state) -> float:
    return float(state.low[2]) - TOP


def _names_under(objects) -> list[str]:
    block = objects["block"]
    under = []
    for name in ROLLER_NAMES:
        p = objects[name].pos
        if block.low[1] <= p[1] <= block.high[1] and block.low[0] <= p[0] <= block.high[0] and p[2] <= TOP + ROLLER_R + UNDER_Z_SLACK:
            under.append(name)
    return under


def rollers_under(objects) -> int:
    """Rollers with their centre inside the block's world-axis bounding box in x and y, and at most UNDER_Z_SLACK above
    lying on the table, inclusive. Positions only: this does not read contact forces."""
    return len(_names_under(objects))


def _slip_step(objects) -> float:
    """How far along y the block moved this step without rolling. A block rolling on rollers moves twice as far as they
    do. So a move is unexplained by as much as it differs from twice the mean move of the rollers now under the block,
    and by all of it when none is under. Capped at the block's own move, so a roller moved about under a still block
    counts nothing. Stationary rollers (carried or slid over) and rollers moving with the block both give the whole
    move; a carried block gets the whole move once no roller is under it."""
    ys = {name: float(objects[name].pos[1]) for name in ("block", *ROLLER_NAMES)}
    last, _memory["last_y"] = _memory["last_y"], ys
    if last is None:
        return 0.0
    moved = ys["block"] - last["block"]
    under = _names_under(objects)
    rollers = sum(ys[name] - last[name] for name in under) / len(under) if under else 0.0
    return min(abs(moved), abs(moved - 2.0 * rollers))


def _in_zone(state) -> bool:
    half = ZONE_LEN / 2
    return (
        float(state.low[1]) >= ZONE_Y - half
        and float(state.high[1]) <= ZONE_Y + half
        and float(state.low[0]) >= LANE_X - ZONE_X_HALF
        and float(state.high[0]) <= LANE_X + ZONE_X_HALF
    )


def block_speed(state) -> float:
    """Upper bound on the speed of any point of the block: its centre's speed plus its spin times its half-diagonal."""
    half_diagonal = float(np.linalg.norm(BLOCK_SIZE)) / 2
    return float(np.linalg.norm(state.lin_vel)) + float(np.linalg.norm(state.ang_vel)) * half_diagonal


def arm_gap(objects) -> float:
    """Smallest distance from any measured arm point to the block or a roller; 0 when the arms were not measured."""
    points = objects.get(ARM_POINTS)
    if points is None:
        return 0.0
    gaps = []
    for name in ("block", *ROLLER_NAMES):
        o = objects[name]
        gaps.append(np.linalg.norm(np.maximum(np.maximum(o.low - points, points - o.high), 0.0), axis=1).min())
    return float(min(gaps))


def hold_steps(objects) -> int:
    """Consecutive control steps spanning HOLD_S at the environment's own rate; HOLD_STEPS at CONTROL_HZ."""
    dt = objects.get(STEP_DT)
    return HOLD_STEPS if dt is None else round(HOLD_S / dt) + 1


def _update(objects) -> None:
    """Advance the memory by one control step; the adapter passes the same states to each hook within a step."""
    if _memory["seen"] is objects:
        return
    trial_step = objects.get(TRIAL_STEP)
    if trial_step is not None and _memory["trial_step"] is not None and trial_step < _memory["trial_step"]:  # a new trial
        _new_trial()
        _memory["trial"] += 1
    _memory["seen"], _memory["trial_step"] = objects, trial_step
    block = objects["block"]
    if _memory["start_y"] is None:
        _memory["start_y"] = float(block.pos[1])
    _memory["slip"] += _slip_step(objects)
    h = _underside(block)
    if (h < ON_TABLE_M or h > LIFTED_M or tilt_deg(block) > 2 * MAX_TILT_DEG or _memory["slip"] > SLIP_MAX_M
            or float(np.linalg.norm(block.lin_vel)) > MAX_BLOCK_SPEED):
        _memory["failed"] = True
    _memory["gap"] = arm_gap(objects)
    _memory["complete"] = (
        not _memory["failed"]
        and _in_zone(block)
        and tilt_deg(block) <= MAX_TILT_DEG
        and rollers_under(objects) >= ROLLERS_NEEDED
        and block_speed(block) < REST_SPEED
        and _memory["gap"] >= CLEAR_M
    )
    _memory["held"] = _memory["held"] + 1 if _memory["complete"] else 0
    _memory["done"] = _memory["held"] >= hold_steps(objects)


def success(objects) -> bool:
    """Every end condition, arms clear included, has held for HOLD_S; the adapter ends the episode on this step."""
    _update(objects)
    return _memory["done"]


def _current_score(block) -> float:
    if _memory["failed"] or tilt_deg(block) > MAX_TILT_DEG:
        return 0.0
    if _memory["done"]:
        return 1.0
    travel = _memory["start_y"] - float(block.pos[1])
    return 0.5 if travel >= TIER_2_M else 0.25 if travel >= TIER_1_M else 0.0


def score(objects) -> float:
    """0 for the rest of the trial once the block has come off the rollers, been raised, tipped over 2 * MAX_TILT_DEG,
    moved too far without rolling or been struck, and 0 on any step it tilts over MAX_TILT_DEG; otherwise 1.0 exactly when success() is
    true, else by how far the block now is from where it was at the first evaluated step."""
    _update(objects)
    return _current_score(objects["block"])


def distance(objects) -> float:
    _log(objects)
    return float(abs(objects["block"].pos[1] - ZONE_Y))


def _log(objects) -> None:
    path = os.environ.get("ROLLER_TRACE")
    if not path:
        return
    _update(objects)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if _memory["steps"] == 0:
            w.writerow(["step", "name", "x", "y", "z", "qx", "qy", "qz", "qw", "speed", "low_z", "failed", "under", "arm_gap", "held",
                        "ang_speed", "trial", "trial_step", "t_s", "slip", "score"])
        under = rollers_under(objects)
        trial_step, dt = objects.get(TRIAL_STEP), objects.get(STEP_DT)
        t_s = "" if trial_step is None or dt is None else round(trial_step * dt, 4)
        for name in ("block", *ROLLER_NAMES):
            o = objects[name]
            w.writerow([_memory["steps"], name, *np.round(o.pos, 5), *np.round(o.quat_xyzw, 4),
                        round(float(np.linalg.norm(o.lin_vel)), 4), round(float(o.low[2]), 5), int(_memory["failed"]), under,
                        round(_memory["gap"], 6), _memory["held"], round(float(np.linalg.norm(o.ang_vel)), 4), _memory["trial"], trial_step, t_s, round(_memory["slip"], 6),
                        _current_score(objects["block"])])
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


def _hold(duration, pos, finger, label="wait") -> Waypoint:
    return Waypoint(duration, pos, finger, label)


def demo():
    """Left arm rolls the block; right arm moves the freed rear roller to the front in between; then both open their
    grippers and move out of the top camera's view of the block, the strip ends and the roller ends. DEMO_MODE selects
    a negative control or a shortcut attempt."""
    mode = os.environ.get("DEMO_MODE", "full")
    if mode in SHORTCUTS:
        return SHORTCUTS[mode]()
    back = START_Y + BLOCK_SIZE[1] / 2
    push_z = TOP + 2 * ROLLER_R + 0.04 + 0.013
    y1 = back + 0.015 - FIRST_PUSH
    y2 = back + 0.015 - TOTAL_PUSH
    left_rest = (0.32, 0.27, TRAVEL_Z)  # far enough that the arms do not meet while the right arm reaches the roller
    # Where the arms end: wide of the top camera's view of the block and strips, and high enough for each wrist camera
    # to take in the block, the rollers and the strip ends from its own side. Chosen by viewing candidates in final frames.
    left_end, left_end_yaw = (LANE_X, 0.045, TOP + 0.24), math.pi / 2
    right_end, right_end_yaw = (LANE_X, -0.30, TOP + 0.24), -math.pi / 2
    if mode == "no_transfer":  # negative controls: push without touching a roller, ...
        y2 = back + 0.015 - float(os.environ.get("PUSH_M", TOTAL_PUSH))  # ... all the way, or stopping short with PUSH_M
        return {"left": [
            Waypoint(2.0, (LANE_X, back + 0.06, TOP + 0.20), FINGER_CLOSED, "above"),
            Waypoint(1.0, (LANE_X, back + 0.04, push_z), FINGER_CLOSED, "down"),
            Waypoint(1.0, (LANE_X, back + 0.015, push_z), FINGER_CLOSED, "touch"),
            Waypoint(12.0, (LANE_X, y2, push_z), FINGER_CLOSED, "push"),
            Waypoint(1.0, (LANE_X, y2, push_z), FINGER_CLOSED, "settle"),
            Waypoint(1.0, (LANE_X, y2 + 0.03, push_z), FINGER_CLOSED, "back off"),
        ]}
    left = [
        Waypoint(2.0, (LANE_X, back + 0.06, TOP + 0.20), FINGER_CLOSED, "above"),
        Waypoint(1.0, (LANE_X, back + 0.04, push_z), FINGER_CLOSED, "down"),
        Waypoint(1.0, (LANE_X, back + 0.015, push_z), FINGER_CLOSED, "touch"),
        Waypoint(5.0, (LANE_X, y1, push_z), FINGER_CLOSED, "push 1"),
        Waypoint(0.5, (LANE_X, y1, push_z), FINGER_CLOSED, "settle"),
        Waypoint(0.8, (LANE_X, y1 + 0.03, push_z), FINGER_CLOSED, "back off"),
        Waypoint(1.0, (LANE_X, y1 + 0.03, TRAVEL_Z), FINGER_CLOSED, "up"),
        Waypoint(1.7, left_rest, FINGER_CLOSED, "clear"),  # t = 13.0
        _hold(15.5, left_rest, FINGER_CLOSED),  # t = 28.5, the right arm moves the roller
        Waypoint(2.0, (LANE_X, y1 + 0.04, TRAVEL_Z), FINGER_CLOSED, "back behind"),
        Waypoint(1.0, (LANE_X, y1 + 0.03, push_z), FINGER_CLOSED, "down 2"),
        Waypoint(1.0, (LANE_X, y1, push_z), FINGER_CLOSED, "touch 2"),
        Waypoint(7.0, (LANE_X, y2, push_z), FINGER_CLOSED, "push 2"),
        Waypoint(0.5, (LANE_X, y2, push_z), FINGER_CLOSED, "settle 2"),
        Waypoint(1.0, (LANE_X, y2 + 0.04, push_z), FINGER_CLOSED, "back off 2"),
        Waypoint(1.4, left_end, GRIP_OPEN, "clear 2", left_end_yaw),  # short enough to arrive within the hold of crossing CLEAR_M
    ]
    if mode == "arm_stays":  # negative control: the same push, but the left finger stays against the block
        left = left[:-2]
    front = START_Y - 0.077 - BLOCK_SIZE[1] / 2  # measured: an 8 cm command rolls the block 7.7 cm
    place_y = front - ROLLER_R - 0.018  # 18 mm of air between the roller and the block's front
    low = TOP + ROLLER_R + 0.001
    right_rest = (0.20, -0.22, TRAVEL_Z)
    right = [
        Waypoint(3.0, right_rest, GRIP_OPEN, "ready"),
        _hold(10.0, right_rest, GRIP_OPEN),  # t = 13.0
        Waypoint(2.2, (GRIP_X, PICK_Y, TRAVEL_Z), GRIP_OPEN, "to roller"),
        Waypoint(2.2, (GRIP_X, PICK_Y, low), GRIP_OPEN, "descend"),
        Waypoint(1.2, (GRIP_X, PICK_Y, low), FINGER_CLOSED, "close"),
        Waypoint(1.8, (GRIP_X, PICK_Y, TRAVEL_Z), FINGER_CLOSED, "lift"),
        Waypoint(2.6, (GRIP_X, place_y, TRAVEL_Z), FINGER_CLOSED, "carry"),
        Waypoint(1.8, (GRIP_X, place_y, low + 0.002), FINGER_CLOSED, "lower"),
        Waypoint(0.9, (GRIP_X, place_y, low + 0.002), GRIP_OPEN, "release"),
        Waypoint(1.4, (GRIP_X, place_y, TRAVEL_Z), GRIP_OPEN, "retreat"),  # t = 27.1
        Waypoint(1.4, right_rest, GRIP_OPEN, "clear"),
        Waypoint(2.0, right_end, GRIP_OPEN, "park", right_end_yaw),
    ]
    return {"left": left, "right": right}


# Shortcut attempts: motions the task does not intend, played as controls to see what they score. Each knob is an
# environment variable, so the receipts of a run say what it did. See review/shortcuts.json for what each one measured.
_BACK = START_Y + BLOCK_SIZE[1] / 2
_FRONT = START_Y - BLOCK_SIZE[1] / 2
_SIDE_Z = TOP + 2 * ROLLER_R + 0.04 + 0.013  # the pad tips at the height of the block's centre
_ROLLER_LOW = TOP + ROLLER_R + 0.001


def _knob(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _pinch() -> tuple[list[Waypoint], list[Waypoint], dict]:
    """Two ways for two grippers to hold a block too wide for either. PINCH_STYLE=faces (0): closed fingers against the
    two side faces, commanded SQUEEZE_M past them since the joints give; along x the rollers do not let the block run
    away from the push. PINCH_STYLE=corners (1): each gripper closes across one of two opposite corners, at 45 degrees
    to its faces, CORNER_IN_M inside it."""
    half_x = BLOCK_SIZE[0] / 2
    if _knob("PINCH_STYLE", 0) == 1:
        inset = _knob("CORNER_IN_M", 0.025) / math.sqrt(2)
        spots = {"left": (LANE_X - half_x + inset, _BACK - inset, math.pi / 4), "right": (LANE_X + half_x - inset, _FRONT + inset, math.pi / 4)}
        ready = {arm: spot[:2] for arm, spot in spots.items()}
        before = FINGER_OPEN
    else:
        pinch = _knob("SQUEEZE_M", 0.09)
        spots = {"left": (LANE_X - half_x - 0.015 + pinch, START_Y, math.pi / 2), "right": (LANE_X + half_x + 0.015 - pinch, START_Y, math.pi / 2)}
        ready = {"left": (LANE_X - half_x - 0.04, START_Y), "right": (LANE_X + half_x + 0.04, START_Y)}
        before = FINGER_CLOSED
    plans = {}
    for arm, (x, y, yaw) in spots.items():
        plans[arm] = [
            Waypoint(2.5, (*ready[arm], TOP + 0.20), before, "above", yaw),
            Waypoint(2.0, (*ready[arm], _SIDE_Z), before, "down", yaw),
            Waypoint(2.0, (x, y, _SIDE_Z), FINGER_CLOSED, "pinch", yaw),
        ]
    return plans["left"], plans["right"], spots


def _carry(left, right, spots, lift: float, shift: float, duration: float) -> None:
    z = _SIDE_Z + lift
    for plan, (x, y, yaw) in ((left, spots["left"]), (right, spots["right"])):
        plan += [Waypoint(2.0, (x, y, z), FINGER_CLOSED, "lift", yaw),
                 Waypoint(duration, (x, y - shift, z), FINGER_CLOSED, "carry", yaw)]


def _set_down(left, right, spots, shift: float) -> None:
    """Lower to the starting height, let go and leave upwards."""
    for plan, (x, y, yaw), away in ((left, spots["left"], -0.08), (right, spots["right"], 0.08)):
        plan += [Waypoint(2.5, (x, y - shift, _SIDE_Z), FINGER_CLOSED, "lower", yaw),
                 Waypoint(1.5, (x + away, y - shift, _SIDE_Z), FINGER_CLOSED, "release", yaw),
                 Waypoint(2.0, (x + away, y - shift, TOP + 0.24), GRIP_OPEN, "leave", yaw)]


def _low_lift():
    """Pinch the block, raise it LIFT_M, carry it SHIFT_M over the rollers, set it back down on them and leave."""
    shift = _knob("SHIFT_M", 0.08)
    left, right, spots = _pinch()
    _carry(left, right, spots, _knob("LIFT_M", 0.03), shift, 6.0)
    _set_down(left, right, spots, shift)
    return {"left": left, "right": right}


def _hold_up():
    """Pinch the block, raise it LIFT_M, carry it into the zone and stay there, holding it up as the trial ends."""
    left, right, spots = _pinch()
    _carry(left, right, spots, _knob("LIFT_M", 0.03), START_Y - ZONE_Y, 12.0)
    return {"left": left, "right": right}


def _shift_roller(y_from: float, y_clear: float, y_to: float, label: str) -> list[Waypoint]:
    """Pinch a roller's end, pull it out from under the block along y, then lift it over and set it down elsewhere."""
    return [
        Waypoint(2.2, (GRIP_X, y_from, TRAVEL_Z), GRIP_OPEN, f"to {label}"),
        Waypoint(2.2, (GRIP_X, y_from, _ROLLER_LOW), GRIP_OPEN, "descend"),
        Waypoint(1.2, (GRIP_X, y_from, _ROLLER_LOW), FINGER_CLOSED, "close"),
        Waypoint(2.5, (GRIP_X, y_clear, _ROLLER_LOW), FINGER_CLOSED, "pull out"),
        Waypoint(1.8, (GRIP_X, y_clear, TRAVEL_Z), FINGER_CLOSED, "lift"),
        Waypoint(2.6, (GRIP_X, y_to, TRAVEL_Z), FINGER_CLOSED, "carry"),
        Waypoint(1.8, (GRIP_X, y_to, _ROLLER_LOW + 0.002), FINGER_CLOSED, "lower"),
        Waypoint(0.9, (GRIP_X, y_to, _ROLLER_LOW + 0.002), GRIP_OPEN, "release"),
        Waypoint(1.4, (GRIP_X, y_to, TRAVEL_Z), GRIP_OPEN, "retreat"),
    ]


def _preplace():
    """Pull the outer two rollers out and set them down in the zone, leaving the block balanced on the middle one; then
    pinch the block, carry it there over the gap and set it down on them. The rolling never happens."""
    y_far, y_near = ZONE_Y - 0.05, ZONE_Y + 0.05
    rear_out = _shift_roller(START_Y + SPACING, _BACK + 0.03, y_near, "rear roller")
    front_out = _shift_roller(START_Y - SPACING, _FRONT - 0.03, y_far, "front roller")
    rear_time, front_time = sum(w.duration for w in rear_out), sum(w.duration for w in front_out)
    left_rest, right_rest = (0.32, 0.27, TRAVEL_Z), (0.20, -0.22, TRAVEL_Z)
    left = [Waypoint(3.0, (0.20, 0.22, TRAVEL_Z), GRIP_OPEN, "ready"), *rear_out, Waypoint(2.0, left_rest, FINGER_CLOSED, "clear"),
            _hold(front_time, left_rest, FINGER_CLOSED)]
    right = [Waypoint(3.0, right_rest, GRIP_OPEN, "ready"), _hold(rear_time + 2.0, right_rest, GRIP_OPEN), *front_out]
    pinch_left, pinch_right, spots = _pinch()
    left += pinch_left
    right += pinch_right
    shift = START_Y - ZONE_Y
    _carry(left, right, spots, _knob("LIFT_M", 0.03), shift, 12.0)
    _set_down(left, right, spots, shift)
    return {"left": left, "right": right}


def _drag_rollers():
    """Push low, so the fingers meet the rear roller and the block's back face together and take both along."""
    y2 = _BACK + 0.015 - _knob("PUSH_M", TOTAL_PUSH)
    z = TOP + PAD_TIP_PAST_TCP + 0.005
    return {"left": [
        Waypoint(2.0, (LANE_X, _BACK + 0.06, TOP + 0.20), FINGER_CLOSED, "above"),
        Waypoint(1.5, (LANE_X, _BACK + 0.04, z), FINGER_CLOSED, "down"),
        Waypoint(1.0, (LANE_X, _BACK + 0.015, z), FINGER_CLOSED, "touch"),
        Waypoint(12.0, (LANE_X, y2, z), FINGER_CLOSED, "push"),
        Waypoint(1.0, (LANE_X, y2, z), FINGER_CLOSED, "settle"),
        Waypoint(1.0, (LANE_X, y2 + 0.04, z), FINGER_CLOSED, "back off"),
        Waypoint(1.5, (LANE_X, y2 + 0.10, TOP + 0.24), GRIP_OPEN, "leave"),
    ]}


def _hold_roller():
    """One arm holds the front roller still, as the only way to stop rollers turning, while the other pushes the block
    along over it."""
    y2 = _BACK + 0.015 - _knob("PUSH_M", TOTAL_PUSH)
    y_held = START_Y - SPACING
    return {
        "right": [
            Waypoint(3.0, (GRIP_X, y_held, TRAVEL_Z), GRIP_OPEN, "to roller"),
            Waypoint(2.2, (GRIP_X, y_held, _ROLLER_LOW), GRIP_OPEN, "descend"),
            Waypoint(1.2, (GRIP_X, y_held, _ROLLER_LOW), FINGER_CLOSED, "close"),
        ],
        "left": [
            Waypoint(2.0, (LANE_X, _BACK + 0.06, TOP + 0.20), FINGER_CLOSED, "above"),
            Waypoint(4.4, (LANE_X, _BACK + 0.04, _SIDE_Z), FINGER_CLOSED, "down"),
            Waypoint(1.0, (LANE_X, _BACK + 0.015, _SIDE_Z), FINGER_CLOSED, "touch"),
            Waypoint(12.0, (LANE_X, y2, _SIDE_Z), FINGER_CLOSED, "push"),
            Waypoint(1.0, (LANE_X, y2, _SIDE_Z), FINGER_CLOSED, "settle"),
            Waypoint(1.0, (LANE_X, y2 + 0.04, _SIDE_Z), FINGER_CLOSED, "back off"),
        ],
    }


def _knock():
    """Strike the block's back face hard at KNOCK_Z above the table, driving KNOCK_M past it in KNOCK_S, and back off."""
    z = TOP + _knob("KNOCK_Z", 0.07) + PAD_TIP_PAST_TCP
    y2 = _BACK + 0.015 - _knob("KNOCK_M", 0.08)
    return {"left": [
        Waypoint(2.0, (LANE_X, _BACK + 0.10, TOP + 0.20), FINGER_CLOSED, "above"),
        Waypoint(1.5, (LANE_X, _BACK + 0.06, z), FINGER_CLOSED, "down"),
        Waypoint(_knob("KNOCK_S", 0.35), (LANE_X, y2, z), FINGER_CLOSED, "strike"),
        Waypoint(1.0, (LANE_X, _BACK + 0.08 - _knob("KNOCK_M", 0.08), z), FINGER_CLOSED, "back off"),
        Waypoint(1.5, (LANE_X, 0.27, TOP + 0.24), GRIP_OPEN, "leave"),
    ]}


def _tip():
    """Press the block's front edge down, tipping it over the front roller, while striking high on its back face."""
    y2 = _BACK + 0.015 - _knob("KNOCK_M", 0.08)
    z = TOP + 0.10 + PAD_TIP_PAST_TCP
    return {
        "right": [
            Waypoint(2.0, (LANE_X, _FRONT - 0.04, TOP + 0.20), FINGER_CLOSED, "above"),
            Waypoint(1.0, (LANE_X, _FRONT + 0.012, TOP + 0.13 + PAD_TIP_PAST_TCP), FINGER_CLOSED, "over the edge"),
            Waypoint(1.5, (LANE_X, _FRONT + 0.012, TOP + 0.11 - _knob("PRESS_M", 0.03) + PAD_TIP_PAST_TCP), FINGER_CLOSED, "press"),
            _hold(2.0, (LANE_X, _FRONT + 0.012, TOP + 0.11 - _knob("PRESS_M", 0.03) + PAD_TIP_PAST_TCP), FINGER_CLOSED),
        ],
        "left": [
            Waypoint(2.0, (LANE_X, _BACK + 0.10, TOP + 0.20), FINGER_CLOSED, "above"),
            Waypoint(2.5, (LANE_X, _BACK + 0.06, z), FINGER_CLOSED, "down"),
            Waypoint(_knob("KNOCK_S", 0.35), (LANE_X, y2, z), FINGER_CLOSED, "strike"),
            Waypoint(1.0, (LANE_X, y2 + 0.08, z), FINGER_CLOSED, "back off"),
        ],
    }


SHORTCUTS = {
    "low_lift": _low_lift,
    "hold_up": _hold_up,
    "preplace": _preplace,
    "drag_rollers": _drag_rollers,
    "hold_roller": _hold_roller,
    "knock": _knock,
    "tip": _tip,
}
