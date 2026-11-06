"""CPU checks of the scoring rules with synthetic states. No simulator needed."""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import roller_block as rb
from yam_arena.task_file import ObjectState


def state(pos, size, tilt_deg=0.0, speed=0.0, spin=0.0):
    half = math.radians(tilt_deg) / 2
    pos, size = np.asarray(pos, float), np.asarray(size, float)
    return ObjectState(pos=pos, quat_xyzw=np.array([math.sin(half), 0.0, 0.0, math.cos(half)]),
                       lin_vel=np.array([0.0, speed, 0.0]), ang_vel=np.array([0.0, 0.0, spin]), low=pos - size / 2, high=pos + size / 2)


ROLLER_SIZE = (rb.ROLLER_LEN, 2 * rb.ROLLER_R, 2 * rb.ROLLER_R)
ROLLER_Z = rb.TOP + rb.ROLLER_R


FAR = 0.30


def world(block_y, rollers_y, block_z=rb.TOP + 0.03 + rb.BLOCK_SIZE[2] / 2, tilt=0.0, speed=0.0, spin=0.0, gap=FAR):
    """The scene; ``gap`` puts one measured arm point that far above the block's top, None leaves the arms unmeasured."""
    objects = {"block": state((rb.LANE_X, block_y, block_z), rb.BLOCK_SIZE, tilt, speed, spin)}
    for name, y in zip(rb.ROLLER_NAMES, rollers_y, strict=True):
        objects[name] = state((rb.LANE_X, y, ROLLER_Z), ROLLER_SIZE)
    if gap is not None:
        objects[rb.ARM_POINTS] = np.array([[rb.LANE_X, block_y, objects["block"].high[2] + gap]])
    return objects


IN_ZONE = (-0.17, -0.11, 0.05)


def settle(steps=rb.HOLD_STEPS, **kwargs):
    """Score ``steps`` consecutive control steps of the block in the zone; returns the last score. Placed there, not
    moved there: the jump from wherever the test last had it does not count as travel."""
    rb._memory["last_y"] = None
    result = None
    for _ in range(steps):
        result = rb.score(world(rb.ZONE_Y, IN_ZONE, **kwargs))
    return result


START = (rb.START_Y - rb.SPACING, rb.START_Y, rb.START_Y + rb.SPACING)


@pytest.fixture(autouse=True)
def fresh():
    rb._new_trial()
    rb._memory["start_y"] = rb.START_Y


def rolled(travel, **kwargs):
    """The block ``travel`` along, with the rollers having rolled half as far beneath it."""
    return world(rb.START_Y - travel, tuple(y - travel / 2 for y in START), **kwargs)


def test_instruction_is_exactly_the_agreed_text():
    assert rb.INSTRUCTION == (
        "In the top camera view, roll the blue block to the right until all of it lies between the two green strips and "
        "within their length, at rest on at least two of the three orange rollers. The rollers are loose: carry a roller "
        "from behind the block to the front of it whenever the block needs one. The block must never touch the table. "
        "At the end, let go and move both arms at least 5 cm away from the block and the rollers."
    )


def test_start_scores_zero():
    assert rb.score(world(rb.START_Y, START)) == 0.0


@pytest.mark.parametrize(("travel", "expected"), [(0.059, 0.0), (0.06, 0.25), (0.149, 0.25), (0.15, 0.5)])
def test_partial_credit_by_travel(travel, expected):
    y = rb.START_Y - travel
    assert rb.score(world(y, (y - 0.03, y, y + 0.03))) == expected


def test_full_score_in_zone_at_rest_on_two_rollers_arms_clear_held():
    assert settle() == 1.0


def test_full_score_only_after_the_hold():
    assert settle(rb.HOLD_STEPS - 1) == 0.5
    assert rb.score(world(rb.ZONE_Y, IN_ZONE)) == 1.0


def test_hold_spans_a_full_second():
    assert (rb.HOLD_STEPS - 1) / rb.CONTROL_HZ == rb.HOLD_S == 1.0


def adapter_ends_at(steps, **kwargs):
    """The step inspect_embodiment._task_step ends the episode on, with its own held count and its own threshold."""
    held = 0
    for i in range(steps):
        objects = {**world(rb.ZONE_Y, IN_ZONE), **kwargs}
        held = held + 1 if rb.success(objects) else 0
        if held >= max(1, round(rb.SUCCESS_HOLD_S * (1 / objects.get(rb.STEP_DT, 1 / rb.CONTROL_HZ)))):
            return i + 1, rb.score(objects)
    return None, rb.score(objects)


def test_the_adapter_ends_the_episode_exactly_when_the_score_is_full():
    assert adapter_ends_at(3 * rb.HOLD_STEPS) == (rb.HOLD_STEPS, 1.0)


@pytest.mark.parametrize(("hz", "steps"), [(10, 11), (20, 21), (40, 41)])
def test_the_hold_is_a_second_at_any_control_rate(hz, steps):
    assert adapter_ends_at(3 * steps, **{rb.STEP_DT: 1 / hz}) == (steps, 1.0)


def test_success_is_not_reported_before_the_hold_is_over():
    objects = world(rb.ZONE_Y, IN_ZONE)
    assert rb.success(objects) is False
    assert rb.score(objects) == 0.5
    assert rb._memory["complete"]


def test_each_step_counts_once_across_the_hooks():
    objects = world(rb.ZONE_Y, IN_ZONE)
    rb.success(objects)
    rb.distance(objects)
    rb.score(objects)
    assert rb._memory["held"] == 1


def test_still_moving_is_not_full_score():
    assert settle(speed=0.05) == 0.5


def test_spinning_about_a_still_centre_is_not_full_score():
    half_diagonal = np.linalg.norm(rb.BLOCK_SIZE) / 2
    assert settle(spin=1.1 * rb.REST_SPEED / half_diagonal) == 0.5


def test_slow_enough_spin_is_at_rest():
    half_diagonal = np.linalg.norm(rb.BLOCK_SIZE) / 2
    assert settle(spin=0.9 * rb.REST_SPEED / half_diagonal) == 1.0


def test_one_roller_under_is_not_full_score():
    for _ in range(rb.HOLD_STEPS):
        result = rb.score(world(rb.ZONE_Y, (-0.14, 0.05, 0.08)))
    assert result == 0.5


def test_sticking_out_of_zone_is_not_full_score():
    y = rb.ZONE_Y + rb.ZONE_LEN / 2 - rb.BLOCK_SIZE[1] / 2 + 0.005
    for _ in range(rb.HOLD_STEPS):
        result = rb.score(world(y, (y - 0.03, y + 0.03, 0.08)))
    assert result == 0.5


@pytest.mark.parametrize("x_off", [-rb.ZONE_X_HALF + rb.BLOCK_SIZE[0] / 2 - 0.002, rb.ZONE_X_HALF - rb.BLOCK_SIZE[0] / 2 + 0.002])
def test_overlapping_a_strip_is_not_full_score(x_off):
    """Past a strip's inner edge, though still short of its centre."""
    for _ in range(rb.HOLD_STEPS):
        objects = world(rb.ZONE_Y, IN_ZONE)
        low, high = objects["block"].low.copy(), objects["block"].high.copy()
        low[0] += x_off
        high[0] += x_off
        objects["block"] = ObjectState(pos=objects["block"].pos, quat_xyzw=objects["block"].quat_xyzw, lin_vel=np.zeros(3),
                                       ang_vel=np.zeros(3), low=low, high=high)
        result = rb.score(objects)
    assert result == 0.5


def test_zone_is_bounded_by_the_strips_inner_edges():
    near, far = rb.ZONE
    assert rb.LANE_X - rb.ZONE_X_HALF == pytest.approx(near.pos[0] + near.size[0] / 2)
    assert rb.LANE_X + rb.ZONE_X_HALF == pytest.approx(far.pos[0] - far.size[0] / 2)


def test_lifted_roller_is_not_under_the_block():
    objects = world(rb.ZONE_Y, IN_ZONE)
    r = objects["roller_0"]
    objects["roller_0"] = ObjectState(pos=r.pos + np.array([0.0, 0.0, rb.UNDER_Z_SLACK + 0.001]), quat_xyzw=r.quat_xyzw,
                                      lin_vel=r.lin_vel, ang_vel=r.ang_vel, low=r.low, high=r.high)
    assert rb.rollers_under(objects) == 1


def test_roller_exactly_at_the_height_limit_is_under_the_block():
    objects = world(rb.ZONE_Y, IN_ZONE)
    r = objects["roller_0"]
    pos = np.array([r.pos[0], r.pos[1], rb.TOP + rb.ROLLER_R + rb.UNDER_Z_SLACK])
    objects["roller_0"] = ObjectState(pos=pos, quat_xyzw=r.quat_xyzw, lin_vel=r.lin_vel, ang_vel=r.ang_vel, low=r.low, high=r.high)
    assert rb.rollers_under(objects) == 2


def test_arm_touching_the_block_at_rest_is_never_full_score():
    assert settle(5 * rb.HOLD_STEPS, gap=0.0) == 0.5
    assert not rb.success(world(rb.ZONE_Y, IN_ZONE, gap=0.0))
    assert adapter_ends_at(5 * rb.HOLD_STEPS, **{rb.ARM_POINTS: np.array([[rb.LANE_X, rb.ZONE_Y, rb.TOP + 0.03 + rb.BLOCK_SIZE[2]]])}) == (None, 0.5)


def test_arm_touching_at_the_last_step_of_the_hold_restarts_it():
    settle(rb.HOLD_STEPS - 1)
    assert not rb.success(world(rb.ZONE_Y, IN_ZONE, gap=0.0))
    assert rb._memory["held"] == 0


@pytest.mark.parametrize(("gap", "expected"), [(rb.CLEAR_M - 0.001, 0.5), (rb.CLEAR_M + 0.001, 1.0)])
def test_clear_distance_is_the_threshold(gap, expected):
    assert settle(gap=gap) == expected


def test_arm_holding_a_roller_clear_of_the_block_is_not_full_score():
    held = (rb.LANE_X + rb.ROLLER_LEN / 2, IN_ZONE[0], rb.TOP + rb.ROLLER_R)
    result = None
    for _ in range(rb.HOLD_STEPS):
        objects = world(rb.ZONE_Y, IN_ZONE)
        objects[rb.ARM_POINTS] = np.array([held])
        result = rb.score(objects)
    assert result == 0.5
    assert np.linalg.norm(np.maximum(np.maximum(objects["block"].low - held, held - objects["block"].high), 0)) > rb.CLEAR_M


def test_unmeasured_arms_are_not_clear():
    assert settle(gap=None) == 0.5


@pytest.mark.parametrize("interrupt", [{"gap": 0.0}, {"speed": 0.05}])
def test_an_interruption_restarts_the_hold(interrupt):
    settle(rb.HOLD_STEPS - 1)
    rb.score(world(rb.ZONE_Y, IN_ZONE, **interrupt))
    assert settle(rb.HOLD_STEPS - 1) == 0.5
    assert rb.score(world(rb.ZONE_Y, IN_ZONE)) == 1.0


def test_failure_stays_failed_even_with_arms_clear_and_held():
    rb.score(world(0.0, START, block_z=rb.TOP + 0.015 + rb.BLOCK_SIZE[2] / 2))
    assert settle() == 0.0
    assert rb._memory["held"] == 0


def test_rolling_back_drops_the_partial_tier():
    assert rb.score(rolled(rb.TIER_2_M)) == 0.5
    assert rb.score(rolled(rb.TIER_1_M)) == 0.25


@pytest.fixture
def stub_env():
    """An environment whose two arms each have every moving link at one point: ``links`` sets which links exist."""
    from yam_arena.analysis import link_hulls

    tips = np.array([0.25, -0.14, 1.0])

    class Tensor:
        def __init__(self, a):
            self.a = np.asarray(a, float)

        def __getitem__(self, i):
            return Tensor(self.a[i])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    def build(links):
        pose = [*tips, 0.0, 0.0, 0.0, 1.0]
        robot = SimpleNamespace(body_names=list(links), data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=Tensor([[pose] * len(links)]))))
        robots = {"left_robot": robot, "right_robot": robot}
        scene = type("Scene", (), {"env_origins": Tensor([[0.0, 0.0, 0.0]]), "__getitem__": lambda self, key: robots[key]})()
        return SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, episode_length_buf=[7], step_dt=0.05))

    return build, [link for link in link_hulls() if link != "base"], tips


def test_scene_hook_adds_arm_points_from_link_poses(stub_env):
    build, links, tips = stub_env
    env = build(links)
    module = SimpleNamespace(
        task_object=lambda spec: spec,
        object_states=lambda e, names: {"block": names},
        STATION_LAMP_INTENSITY=100.0,
        STATION_DOME_INTENSITY=100.0,
    )
    rb._extend_scene(module)  # twice must not stack
    rb._extend_scene(module)
    out = module.object_states(env, "names")
    assert out["block"] == "names"
    assert module.STATION_LAMP_INTENSITY == 100.0  # the adapter's lighting is left alone
    assert module.STATION_DOME_INTENSITY == 100.0
    points = out[rb.ARM_POINTS]
    assert points.shape[1] == 3
    assert np.allclose(points.mean(axis=0)[:2], tips[:2], atol=0.2)
    assert abs(points[:, 2].min() - tips[2]) < 0.3
    assert out[rb.TRIAL_STEP] == 7
    assert out[rb.STEP_DT] == 0.05
    assert set(out[rb.ARM_POSES]) == {f"{arm}/{link}" for arm in ("left_robot", "right_robot") for link in links}


def test_trace_records_slip_and_the_score_of_each_step(tmp_path, monkeypatch):
    trace = tmp_path / "t.csv"
    monkeypatch.setenv("ROLLER_TRACE", str(trace))
    monkeypatch.setitem(rb._memory, "steps", 0)
    rb.distance(rolled(0.0))
    rb.distance(world(rb.START_Y - 0.07, START))
    blocks = [r for r in csv.DictReader(open(trace)) if r["name"] == "block"]
    assert [float(r["slip"]) for r in blocks] == [0.0, pytest.approx(0.07)]
    assert [float(r["score"]) for r in blocks] == [0.0, 0.0]


def test_a_missing_arm_link_means_the_arms_are_not_clear(stub_env):
    build, links, _ = stub_env
    points, poses = rb._arm_state(build(links[1:]))
    assert points is None
    assert len(poses) == 2 * (len(links) - 1)
    assert settle(gap=None) == 0.5


def test_trace_lets_arm_gap_be_recomputed(tmp_path, monkeypatch):
    trace = tmp_path / "t.csv"
    monkeypatch.setenv("ROLLER_TRACE", str(trace))
    monkeypatch.setitem(rb._memory, "steps", 0)
    monkeypatch.setitem(rb._link_vertices, "tip", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]))
    pose = np.eye(4)
    pose[:3, 3] = [rb.LANE_X, rb.ZONE_Y, 1.2]
    objects = world(rb.ZONE_Y, IN_ZONE)
    objects[rb.ARM_POSES] = {"left_robot/tip": pose}
    rb.distance(objects)
    rows = list(csv.DictReader(open(trace.with_suffix(".arms.csv"))))
    assert [(r["step"], r["arm"], r["link"], float(r["z"])) for r in rows] == [("0", "left_robot", "tip", 1.2)]
    assert float(rows[0]["r00"]) == 1.0
    assert len(list(csv.DictReader(open(trace.with_suffix(".hulls.csv"))))) >= 2
    main = list(csv.DictReader(open(trace)))
    assert {"arm_gap", "held", "ang_speed", "trial", "trial_step", "t_s"} <= set(main[0])


def test_trace_records_the_trial_and_the_time(tmp_path, monkeypatch):
    trace = tmp_path / "t.csv"
    monkeypatch.setenv("ROLLER_TRACE", str(trace))
    monkeypatch.setitem(rb._memory, "steps", 0)
    monkeypatch.setitem(rb._memory, "trial", 0)
    for step in (40, 41, 1):  # the environment's count falls back at a reset
        rb.distance({**world(rb.ZONE_Y, IN_ZONE), rb.TRIAL_STEP: step, rb.STEP_DT: 0.05})
    blocks = [r for r in csv.DictReader(open(trace)) if r["name"] == "block"]
    assert [(r["trial"], r["trial_step"], float(r["t_s"])) for r in blocks] == [("0", "40", 2.0), ("0", "41", 2.05), ("1", "1", 0.05)]
    assert [r["step"] for r in blocks] == ["0", "1", "2"]


def _demo(monkeypatch, mode):
    monkeypatch.setenv("DEMO_MODE", mode)
    return rb.demo()


def test_demo_ends_with_both_arms_away_and_both_grippers_open(monkeypatch):
    plan = _demo(monkeypatch, "full")
    y_block_back = rb.ZONE_Y + rb.BLOCK_SIZE[1] / 2 + 0.03
    assert plan["left"][-1].pos[1] > y_block_back
    assert plan["left"][-1].finger == rb.GRIP_OPEN
    assert plan["right"][-1].pos[1] < rb.ZONE_Y - rb.BLOCK_SIZE[1] / 2
    assert plan["right"][-1].finger == rb.GRIP_OPEN
    assert min(plan["left"][-1].pos[2], plan["right"][-1].pos[2]) > rb.TOP + 0.03 + rb.BLOCK_SIZE[2] + rb.CLEAR_M


def test_arm_stays_control_is_the_full_push_without_the_retreat(monkeypatch):
    full, stays = _demo(monkeypatch, "full"), _demo(monkeypatch, "arm_stays")
    assert [w.label for w in stays["left"]] == [w.label for w in full["left"][:-2]]
    assert stays["left"][-1].label == "settle 2"
    assert stays["right"] == full["right"]


def test_task_json_matches_this_file():
    task = json.loads((Path(rb.__file__).parent / "task.json").read_text())
    assert task["instruction"] == rb.INSTRUCTION
    half = rb.ZONE_LEN / 2
    for text in (
        f"y {rb.ZONE_Y - half:.2f} to {rb.ZONE_Y + half:.2f} m",
        f"x {rb.LANE_X - rb.ZONE_X_HALF:.3f} to {rb.LANE_X + rb.ZONE_X_HALF:.3f} m",
        f"at most {rb.MAX_TILT_DEG:.0f} degrees",
        f"under {rb.REST_SPEED * 100:.0f} cm/s",
        f"At least {rb.ROLLERS_NEEDED} rollers",
        f"at most {rb.UNDER_Z_SLACK * 1000:.0f} mm above",
        f"at least {rb.CLEAR_M * 100:.0f} cm from",
        f"on {rb.HOLD_STEPS} consecutive control steps",
        f"{rb.HOLD_S:.0f} s from the first to the last at {rb.CONTROL_HZ} Hz",
        f"0.5 if the block's centre is now at least {rb.TIER_2_M * 100:.0f} cm",
        f"0.25 at {rb.TIER_1_M * 100:.0f} cm",
        f"under {rb.ON_TABLE_M * 100:.0f} cm or over {rb.LIFTED_M * 100:.1f} cm",
        f"more than {rb.SLIP_MAX_M * 100:.0f} cm",
        f"faster than {rb.MAX_BLOCK_SPEED * 100:.0f} cm/s",
        f"tilts over {2 * rb.MAX_TILT_DEG:.0f} degrees",
    ):
        assert text in task["scoring"], text


def test_touching_the_table_fails_for_good():
    assert rb.score(world(0.0, START, block_z=rb.TOP + 0.015 + rb.BLOCK_SIZE[2] / 2)) == 0.0
    assert settle() == 0.0


def test_lifting_fails_for_good():
    assert rb.score(world(0.0, START, block_z=rb.TOP + 0.05 + rb.BLOCK_SIZE[2] / 2)) == 0.0
    assert settle() == 0.0


def test_tipped_over_fails_for_good():
    assert rb.score(world(0.0, START, tilt=11.0)) == 0.0
    assert settle() == 0.0


def test_slight_tilt_scores_zero_but_can_recover():
    assert rb.score(world(0.0, START, tilt=6.0)) == 0.0
    assert not rb._memory["failed"]
    assert settle() == 1.0


def test_putting_the_layout_back_does_not_clear_a_failure():
    def stepped(step, *args, **kwargs):
        return {**world(*args, **kwargs), rb.TRIAL_STEP: step}

    rb.score(stepped(1, 0.0, START, block_z=rb.TOP + 0.015 + rb.BLOCK_SIZE[2] / 2))
    assert rb._memory["failed"]
    assert rb.score(stepped(2, rb.START_Y, START)) == 0.0
    assert rb._memory["failed"]
    for i in range(rb.HOLD_STEPS):
        assert rb.score(stepped(3 + i, rb.ZONE_Y, IN_ZONE)) == 0.0


def test_only_a_falling_step_count_is_a_new_trial():
    """The written rule: a new trial is the environment's step count falling back. Staying level is not one."""

    def stepped(step, *args, **kwargs):
        return {**world(*args, **kwargs), rb.TRIAL_STEP: step}

    rb.score(stepped(40, 0.0, START, block_z=rb.TOP + 0.015 + rb.BLOCK_SIZE[2] / 2))
    rb.score(stepped(40, rb.START_Y, START))
    assert rb._memory["failed"]
    rb.score(stepped(39, rb.START_Y, START))
    assert not rb._memory["failed"]


def test_a_new_trial_forgets_the_slip_and_the_start():
    def stepped(step, *args, **kwargs):
        return {**world(*args, **kwargs), rb.TRIAL_STEP: step}

    rb.score(stepped(11, rb.START_Y, START))
    rb.score(stepped(12, rb.START_Y - 0.05, START))
    assert rb._memory["failed"]
    assert rb._memory["slip"] == pytest.approx(0.05)
    rb.score(stepped(11, rb.START_Y + 0.01, START))
    assert rb._memory["slip"] == 0.0
    assert rb._memory["start_y"] == pytest.approx(rb.START_Y + 0.01)


def test_tiers_are_measured_from_where_the_block_actually_started():
    rb._memory["start_y"] = None
    y0 = rb.START_Y - 0.03
    assert rb.score(world(y0, START)) == 0.0
    assert rb._memory["start_y"] == pytest.approx(y0)
    assert rb.score(world(y0 - rb.TIER_1_M + 0.001, tuple(y - (rb.TIER_1_M - 0.001) / 2 for y in START))) == 0.0
    assert rb.score(world(y0 - rb.TIER_1_M, tuple(y - rb.TIER_1_M / 2 for y in START))) == 0.25


# -- shortcuts: each is also played in the simulator, see review/shortcuts.json ---------------------------------------


def carry(travels, rollers_at, **kwargs):
    """A block moved through ``travels``, with ``rollers_at(travel)`` giving the rollers' y; the last score."""
    result = None
    for travel in travels:
        result = rb.score(world(rb.START_Y - travel, rollers_at(travel), **kwargs))
    return result


def along(total, step=0.005):
    return np.arange(0.0, total + step / 2, step)


def test_a_rolling_block_slips_nothing():
    assert carry(along(rb.TIER_1_M), lambda t: tuple(y - t / 2 for y in START)) == 0.25
    assert rb._memory["slip"] == pytest.approx(0.0)
    assert not rb._memory["failed"]


@pytest.mark.parametrize("lift", [0.0, 0.002, rb.LIFTED_M - rb.ROLLER_R * 2 - 0.0001])
def test_carried_over_stationary_rollers_scores_nothing_at_any_height_in_the_band(lift):
    z = rb.TOP + 0.03 + lift + rb.BLOCK_SIZE[2] / 2
    assert carry(along(rb.TIER_2_M), lambda t: START, block_z=z) == 0.0
    assert rb._memory["failed"]
    assert rb._memory["slip"] > rb.SLIP_MAX_M


def test_slid_or_carried_up_to_the_allowance_does_not_fail_past_it_does():
    assert carry(along(rb.SLIP_MAX_M - 0.0005, 0.0005), lambda t: START) == 0.0
    assert not rb._memory["failed"]
    assert rb.score(world(rb.START_Y - rb.SLIP_MAX_M - 0.001, START)) == 0.0
    assert rb._memory["failed"]


def test_rollers_dragged_along_with_the_block_score_nothing():
    assert carry(along(rb.TIER_2_M), lambda t: tuple(y - t for y in START)) == 0.0
    assert rb._memory["failed"]


def test_one_roller_dragged_with_two_rolling_scores_nothing():
    """The low push: its fingers take the rear roller along at the block's speed, the other two roll."""
    assert carry(along(rb.TIER_2_M), lambda t: (START[0] - t / 2, START[1] - t / 2, START[2] - t)) == 0.0
    assert rb._memory["failed"]


def test_one_roller_held_still_with_two_rolling_scores_nothing_past_the_first_tier():
    assert carry(along(rb.TIER_2_M), lambda t: (START[0], START[1] - t / 2, START[2] - t / 2)) == 0.0
    assert rb._memory["failed"]


def test_set_down_on_rollers_placed_in_the_zone_is_not_full_score():
    """Carried over with no roller under it, then lowered onto two put there beforehand: the end state alone is perfect."""
    middle = START[1]
    for travel in along(rb.START_Y - rb.ZONE_Y):
        rb.score(world(rb.START_Y - travel, (IN_ZONE[0], middle, IN_ZONE[1])))
    assert rb._memory["failed"]
    for _ in range(rb.HOLD_STEPS):
        result = rb.score(world(rb.ZONE_Y, (IN_ZONE[0], middle, IN_ZONE[1])))
    assert result == 0.0
    assert not rb._memory["done"]


def test_held_up_in_the_zone_scores_nothing():
    z = rb.TOP + 0.034 + rb.BLOCK_SIZE[2] / 2
    assert carry(along(rb.START_Y - rb.ZONE_Y), lambda t: START, block_z=z, gap=0.0) == 0.0


@pytest.mark.parametrize(("underside", "fails"), [(rb.LIFTED_M - 0.0001, False), (rb.LIFTED_M + 0.0005, True), (rb.ON_TABLE_M + 0.0001, False), (rb.ON_TABLE_M - 0.0005, True)])
def test_the_height_band_edges(underside, fails):
    rb.score(world(rb.START_Y, START, block_z=rb.TOP + underside + rb.BLOCK_SIZE[2] / 2))
    assert rb._memory["failed"] == fails


@pytest.mark.parametrize(("speed", "fails"), [(rb.MAX_BLOCK_SPEED - 0.001, False), (rb.MAX_BLOCK_SPEED + 0.001, True)])
def test_a_struck_block_fails_for_good(speed, fails):
    rb.score(world(rb.START_Y, START, speed=speed))
    assert rb._memory["failed"] == fails
    assert settle() == (0.0 if fails else 1.0)


def test_a_roller_stood_on_another_is_not_under_the_block():
    objects = world(rb.ZONE_Y, IN_ZONE)
    r = objects["roller_0"]
    objects["roller_0"] = ObjectState(pos=r.pos + np.array([0.0, 0.0, 2 * rb.ROLLER_R]), quat_xyzw=r.quat_xyzw, lin_vel=r.lin_vel, ang_vel=r.ang_vel, low=r.low, high=r.high)
    assert rb.rollers_under(objects) == 1


def test_a_roller_moved_about_under_a_still_block_is_not_slip():
    assert carry([0.0], lambda t: START) == 0.0
    for y in (0.06, 0.05, 0.04, 0.03):
        rb.score(world(rb.START_Y, (y, START[1], START[2])))
    assert rb._memory["slip"] == 0.0


def test_a_block_that_rolled_gets_there_through_a_transfer_and_scores_full():
    """The intended sequence in small steps: roll, carry the rear roller over while the block is still, roll again."""
    r0, r1, r2 = START
    y, first = rb.START_Y, 0.08
    for _ in range(16):
        y, r0, r1, r2 = y - 0.005, r0 - 0.0025, r1 - 0.0025, r2 - 0.0025
        rb.score(world(y, (r0, r1, r2)))
    assert rb.START_Y - y == pytest.approx(first)
    r2 = y - rb.BLOCK_SIZE[1] / 2 - 0.033
    rb.score(world(y, (r0, r1, r2)))
    while y > rb.ZONE_Y:
        y, r0, r1, r2 = y - 0.005, r0 - 0.0025, r1 - 0.0025, r2 - 0.0025
        rb.score(world(y, (r0, r1, r2)))
    assert rb._memory["slip"] < rb.SLIP_MAX_M
    assert not rb._memory["failed"]
    for _ in range(rb.HOLD_STEPS):
        result = rb.score(world(y, (r0, r1, r2)))
    assert result == 1.0


def test_every_shortcut_mode_is_a_plan_the_policy_can_play(monkeypatch):
    for mode in rb.SHORTCUTS:
        plan = _demo(monkeypatch, mode)
        assert plan
        assert set(plan) <= {"left", "right"}
        for waypoints in plan.values():
            assert all(w.duration > 0 for w in waypoints)


def test_the_default_demo_is_not_a_shortcut(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert "push 2" in [w.label for w in rb.demo()["left"]]


def test_the_environment_resetting_clears_a_failure():
    """A later trial starts with the environment's step count back down, and its first step is scored afresh."""

    def stepped(step, *args, **kwargs):
        return {**world(*args, **kwargs), rb.TRIAL_STEP: step}

    rb.score(stepped(40, 0.0, START, block_z=rb.TOP + 0.015 + rb.BLOCK_SIZE[2] / 2))
    assert rb._memory["failed"]
    rb.score(stepped(41, rb.START_Y, START))
    assert rb._memory["failed"]
    assert rb.score(stepped(40, rb.START_Y, START)) == 0.0
    assert not rb._memory["failed"]
