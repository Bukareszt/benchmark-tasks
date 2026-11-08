"""Scoring tests for drum_ramp.py with synthetic states; no simulator. Run: python -m pytest test_scoring.py"""
import math

import numpy as np
import pytest

import drum_ramp as t
from yam_arena.task_file import ObjectState

TOP = t.TOP


def quat_about_x(deg):
    h = math.radians(deg) / 2
    return np.array([math.sin(h), 0.0, 0.0, math.cos(h)])


def quat_about_y(deg):
    h = math.radians(deg) / 2
    return np.array([0.0, math.sin(h), 0.0, math.cos(h)])


def quat_about_z(deg):
    h = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(h), math.cos(h)])


def drum(x, z=None, y=0.0, quat=None, vel=(0, 0, 0), spin=(0, 0, 0), yaw_deg=0.0):
    if z is None:
        z = t.rolling_height(x, y)
    q = quat if quat is not None else quat_about_z(yaw_deg)
    pos = np.array([x, y, z], float)
    c, s = abs(math.cos(math.radians(yaw_deg))), abs(math.sin(math.radians(yaw_deg)))
    half = np.array([t.DRUM_R * c + t.DRUM_LEN / 2 * s, t.DRUM_LEN / 2 * c + t.DRUM_R * s, t.DRUM_R])
    return ObjectState(pos, q, np.array(vel, float), np.array(spin, float), pos - half, pos + half)


def fixed(name):
    o = {"ramp": t.RAMP, "deck": t.DECK, "stop": t.STOP}[name]
    pos = np.array(o.pos)
    half = np.array(o.size) / 2
    return ObjectState(pos, np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3), pos - half, pos + half)


def states(d, tips=None, step=None, points=None):
    far = {"left": np.array([0.1, 0.25, TOP + 0.3]), "right": np.array([0.1, -0.25, TOP + 0.3])}
    tips = tips or far
    if points is None:
        points = np.array(list(tips.values()))  # the arms stand in for their fingertips unless given
    s = {"drum": d, "ramp": fixed("ramp"), "deck": fixed("deck"), "stop": fixed("stop"), t.TIPS: tips, t.ARM_POINTS: points, t.STEP_DT: 0.05}
    if step is not None:
        s[t.TRIAL_STEP] = step
    return s


@pytest.fixture(autouse=True)
def fresh():
    t._memory.update(seen=None, trial_step=None, trial=0, lifted=False, start_z=None, steps=0)


DECK_C = (t.RAMP_X1 + t.ON_DECK_PAST_EDGE + t.STOP_X0) / 2


def test_surface_profile():
    assert t.surface_z(0.10, 0) == TOP
    assert t.surface_z(t.RAMP_X0, 0) == pytest.approx(TOP)
    assert t.surface_z((t.RAMP_X0 + t.RAMP_X1) / 2, 0) == pytest.approx(TOP + t.DECK_H / 2)
    assert t.surface_z(t.RAMP_X1 - 1e-9, 0) == pytest.approx(TOP + t.DECK_H)
    assert t.surface_z(0.50, 0) == TOP + t.DECK_H
    assert t.surface_z(0.50, 0.30) == TOP  # beside the deck
    assert t.rolling_height((t.RAMP_X0 + t.RAMP_X1) / 2, 0) == pytest.approx(TOP + t.DECK_H / 2 + t.DRUM_R / math.cos(t.SLOPE))


def test_start_state_scores_zero_and_is_not_success():
    s = states(drum(t.DRUM_X0))
    assert not t.success(s) and t.score(s) == 0.0


def test_on_the_deck_still_and_clear_is_success():
    s = states(drum(DECK_C))
    assert t.success(s) and t.score(s) == 1.0


def test_on_the_deck_against_the_stop_is_success():
    s = states(drum(t.STOP_X0 - t.DRUM_R))
    assert t.success(s)


def test_off_the_deck_centre_line_is_not_on_the_deck():
    start()
    assert t.success(states(drum(DECK_C, y=0.03)))
    assert not t.success(states(drum(DECK_C, y=0.05)))


def test_on_the_deck_but_moving_or_held_is_half():
    assert t.score(states(drum(DECK_C, vel=(0.05, 0, 0)))) == 0.5
    near = {"left": np.array([DECK_C - t.DRUM_R - 0.01, 0.1, TOP + t.DECK_H + 0.05]), "right": np.array([0.1, -0.25, TOP + 0.3])}
    s = states(drum(DECK_C), tips=near)
    assert not t.success(s) and t.score(s) == 0.5
    edge = {"left": np.array([DECK_C - t.DRUM_R - t.CLEAR_M - 0.001, 0.1, TOP + t.DECK_H + 0.05]), "right": near["right"]}
    assert t.success(states(drum(DECK_C), tips=edge))


def start():
    t.score(states(drum(t.DRUM_X0), step=11))


def test_partly_over_the_ramp_edge_is_not_on_the_deck():
    start()
    s = states(drum(t.RAMP_X1 + t.ON_DECK_PAST_EDGE - 0.01, z=TOP + t.DECK_H + t.DRUM_R))
    assert not t.success(s) and t.score(s) == 0.25  # higher than at the start, so partway


def test_skewed_drum_on_the_deck_is_not_on_the_deck():
    start()
    assert t.success(states(drum(DECK_C, yaw_deg=10)))
    assert not t.success(states(drum(DECK_C, yaw_deg=20)))
    assert t.score(states(drum(DECK_C, yaw_deg=20))) == 0.25


def test_partway_up_the_ramp_is_a_quarter():
    start()
    s = states(drum((t.RAMP_X0 + t.RAMP_X1) / 2))  # held on the ramp
    assert t.score(s) == 0.25
    assert t.score(states(drum(t.RAMP_X0 + 0.02))) == 0.0  # less than 2 cm up


def test_lifted_latches_zero_for_the_trial():
    t.score(states(drum(t.DRUM_X0), step=11))
    t.score(states(drum(0.26, z=TOP + t.DRUM_R + 0.03), step=12))  # raised 3 cm
    assert t._memory["lifted"]
    assert t.score(states(drum(DECK_C), step=13)) == 0.0
    assert not t.success(states(drum(DECK_C), step=14))
    # a new trial (step count falls back) clears it
    assert t.score(states(drum(DECK_C), step=11)) == 1.0


def test_rolling_up_the_ramp_is_not_lifted():
    for i, x in enumerate(np.linspace(t.DRUM_X0, DECK_C, 60)):
        t.score(states(drum(x), step=11 + i))
    assert not t._memory["lifted"]


def test_drum_resting_on_the_deck_edge_from_the_ramp_is_not_lifted():
    # touching the deck's top edge with its centre still over the ramp
    for x in (t.RAMP_X1 - 0.03, t.RAMP_X1 - 0.02, t.RAMP_X1 - 0.01):
        z = TOP + t.DECK_H + math.sqrt(t.DRUM_R**2 - (t.RAMP_X1 - x) ** 2)
        assert z <= t.rolling_height(x, 0) + t.LIFT_M


def test_lift_threshold_is_above_table_to_ramp_transition():
    # the drum touching the ramp's foot edge while its centre is still over the table
    for x in np.linspace(t.RAMP_X0 - t.DRUM_R * math.sin(t.SLOPE), t.RAMP_X0, 10):
        # resting on the ramp plane: distance from centre to the plane equals R
        z = TOP + (x - t.RAMP_X0) * math.tan(t.SLOPE) + t.DRUM_R / math.cos(t.SLOPE)
        assert z <= t.rolling_height(x, 0) + t.LIFT_M


def test_underside_ignores_the_rolled_bounding_box():
    d = drum(DECK_C)
    rolled = ObjectState(d.pos, quat_about_y(165), d.lin_vel, d.ang_vel, d.low - 0.03, d.high + 0.03)
    assert t.underside(rolled) == pytest.approx(TOP + t.DECK_H)
    assert t.success(states(rolled))


def test_arm_body_against_the_drum_is_not_clear_even_with_the_fingertips_away():
    start()
    d = drum(DECK_C)
    far_tips = {"left": np.array([DECK_C - 0.16, 0.09, TOP + 0.003]), "right": np.array([DECK_C - 0.16, -0.09, TOP + 0.003])}
    body = np.array([[DECK_C - t.DRUM_R - 0.001, 0.09, TOP + t.DECK_H + t.DRUM_R], far_tips["left"], far_tips["right"]])
    s = states(d, tips=far_tips, points=body)
    assert t.drum_distance(d, far_tips["left"]) > t.CLEAR_M  # the fingertip rule alone would pass
    assert not t.success(s) and t.score(s) == 0.5
    assert t.score(states(d, tips=far_tips)) == 1.0


def test_missing_arm_points_is_not_clear():
    start()
    s = states(drum(DECK_C))
    s[t.ARM_POINTS] = None
    assert not t.success(s) and t.score(s) == 0.5


def test_drum_distance():
    d = drum(DECK_C)
    assert t.drum_distance(d, d.pos) == 0.0
    assert t.drum_distance(d, d.pos + np.array([0, 0, t.DRUM_R + 0.03])) == pytest.approx(0.03)
    assert t.drum_distance(d, d.pos + np.array([0, t.DRUM_LEN / 2 + 0.02, 0])) == pytest.approx(0.02)
    assert t.drum_distance(d, d.pos + np.array([t.DRUM_R + 0.02, 0, 0])) == pytest.approx(0.02)


def test_axis_error():
    assert t.axis_error_deg(drum(0.3)) == pytest.approx(0.0)
    assert t.axis_error_deg(drum(0.3, yaw_deg=25)) == pytest.approx(25.0)
    assert t.axis_error_deg(drum(0.3, quat=quat_about_x(30))) == pytest.approx(30.0)


def test_fallen_off_the_side_scores_zero_without_lifting():
    s = states(drum((t.RAMP_X0 + t.RAMP_X1) / 2, y=0.25, z=TOP + t.DRUM_R))
    assert t.score(s) == 0.0 and not t._memory["lifted"]


def test_scene_geometry():
    assert t.DRUM_R * 2 > 0.094
    assert t.RAMP.size[0] == pytest.approx(math.hypot(t.RAMP_X1 - t.RAMP_X0, t.DECK_H))
    assert t.STOP.pos[2] - t.STOP.size[2] / 2 == pytest.approx(TOP + t.DECK_H)
    assert t.DECK.pos[0] + t.DECK.size[0] / 2 == pytest.approx(t.DECK_X1)
    assert t.STOP_X0 - t.DRUM_R > t.RAMP_X1 + t.ON_DECK_PAST_EDGE + 0.03  # room for the drum on the deck
    assert t.DECK_X1 - t.RAMP_X1 > 2 * t.DRUM_R  # the whole drum fits on the deck
