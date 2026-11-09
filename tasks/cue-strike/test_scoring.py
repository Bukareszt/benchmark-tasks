"""Scoring tests for cue_strike.py with synthetic states; no simulator. Run: python -m pytest test_scoring.py"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

import cue_strike as t
from yam_arena.task_file import ObjectState

BALL_Z = t.FLOOR + t.BALL_R


def quat_z(deg):
    h = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(h), math.cos(h)])


def ball(x, y, z=BALL_Z, vel=(0, 0, 0)):
    p = np.array([x, y, z], float)
    return ObjectState(p, np.array([0, 0, 0, 1.0]), np.array(vel, float), np.zeros(3), p - t.BALL_R, p + t.BALL_R)


def stick(x=t.STICK0[0], y=t.STICK0[1], yaw_deg=0.0, z=t.FLOOR + t.STICK_W / 2):
    p = np.array([x, y, z], float)
    return ObjectState(p, quat_z(yaw_deg), np.zeros(3), np.zeros(3), p - 0.2, p + 0.2)


FAR_ARMS = np.array([[0.10, 0.30, t.TOP + 0.3], [0.10, -0.30, t.TOP + 0.3]])


def states(white=None, red=None, stk=None, arms=FAR_ARMS, step=None):
    s = {
        "white": white or ball(*t.WHITE0),
        "red": red or ball(*t.RED0),
        "stick": stk or stick(),
        t.ARM_POINTS: arms,
        t.STEP_DT: 0.05,
    }
    if step is not None:
        s[t.TRIAL_STEP] = step
    return s


IN_GOAL = (t.GOAL_X[1] - t.BALL_R, t.GOAL_Y)


@pytest.fixture(autouse=True)
def fresh():
    t._memory.update(seen=None, trial_step=None, trial=0, foul=None, red0=None, steps=0)


def start():
    t.score(states(step=11))


def test_start_scores_zero():
    s = states()
    assert t.score(s) == 0.0 and not t.success(s)


def test_red_at_rest_in_the_goal_is_full_score():
    start()
    s = states(red=ball(*IN_GOAL))
    assert t.success(s) and t.score(s) == 1.0


def test_red_still_rolling_in_the_goal_is_not_full():
    start()
    s = states(red=ball(*IN_GOAL, vel=(0.05, 0, 0)))
    assert not t.success(s) and t.score(s) == 0.5


def test_red_in_the_mouth_less_than_a_radius_inside_is_not_in():
    start()
    assert not t.in_goal(ball(t.GOAL_MOUTH_X + t.BALL_R - 0.002, t.GOAL_Y))
    assert t.in_goal(ball(t.GOAL_MOUTH_X + t.BALL_R + 0.002, t.GOAL_Y))


def test_red_beside_the_goal_is_not_in():
    assert not t.in_goal(ball(t.GOAL_C[0], t.GOAL_YS[0] - 0.02))
    assert not t.in_goal(ball(t.GOAL_C[0], t.GOAL_YS[1] + 0.02))


def test_red_resting_on_a_wall_top_is_not_in():
    assert not t.in_goal(ball(*IN_GOAL, z=t.FLOOR + t.WALL_H + t.BALL_R))


def test_partial_credit():
    start()
    assert t.score(states(red=ball(t.RED0[0] + 0.04, t.RED0[1]))) == 0.0
    assert t.score(states(red=ball(t.RED0[0] + 0.06, t.RED0[1]))) == 0.25
    near = (t.GOAL_C[0] - t.NEAR_M + 0.005, t.GOAL_C[1])
    assert t.score(states(red=ball(*near))) == 0.5


def test_stick_over_the_line_is_a_foul_for_the_rest_of_the_trial():
    start()
    over = t.LINE_X - t.LINE_W / 2 - t.STICK_LEN / 2 + 0.002
    assert t.score(states(stk=stick(x=over))) == 0.0
    assert t._memory["foul"] == "stick_over_line"
    assert t.score(states(red=ball(*IN_GOAL))) == 0.0
    assert not t.success(states(red=ball(*IN_GOAL)))


def test_stick_just_behind_the_line_is_fine():
    start()
    x = t.LINE_X - t.LINE_W / 2 - t.STICK_LEN / 2 - 0.002
    t.score(states(stk=stick(x=x)))
    assert t._memory["foul"] is None


def test_turned_stick_front_corner_counts():
    s = stick(x=t.LINE_X - t.LINE_W / 2 - t.STICK_LEN / 2 - 0.001, yaw_deg=20)
    a = math.radians(20)
    expect = s.pos[0] + t.STICK_LEN / 2 * math.cos(a) + t.STICK_W / 2 * math.sin(a)
    assert t.stick_front_x(s) == pytest.approx(expect)


def test_arm_over_the_line_is_a_foul():
    start()
    arms = np.vstack([FAR_ARMS, [[t.LINE_X - t.LINE_W / 2 + 0.001, 0.25, t.TOP + 0.3]]])
    t.score(states(arms=arms))
    assert t._memory["foul"] == "arm_over_line"


@pytest.mark.parametrize("name", ["white", "red"])
def test_arm_near_a_ball_is_a_foul(name):
    start()
    b = ball(*(t.WHITE0 if name == "white" else (0.40, 0.0)))  # red brought back behind the line for the test
    near = b.pos + np.array([-(t.BALL_R + t.ARM_TOUCH_M - 0.001), 0, 0])
    s = states(**{name: b}, arms=np.vstack([FAR_ARMS, near]))
    t.score(s)
    assert f"arm_touched_{name}" in t._memory["foul"]


def test_arm_just_outside_the_touch_distance_is_fine():
    start()
    b = ball(*t.WHITE0)
    near = b.pos + np.array([-(t.BALL_R + t.ARM_TOUCH_M + 0.001), 0, 0])
    t.score(states(white=b, arms=np.vstack([FAR_ARMS, near])))
    assert t._memory["foul"] is None


def test_unmeasured_arms_are_a_foul():
    start()
    t.score(states(arms=None))
    assert t._memory["foul"] is not None


def test_stick_touching_the_red_ball_is_a_foul():
    start()
    red = ball(0.30, 0.10)  # rolled back behind the line
    stk = stick(x=red.pos[0] - t.BALL_R - t.STICK_LEN / 2 - 0.001, y=0.10)
    t.score(states(red=red, stk=stk))
    assert t._memory["foul"] == "stick_touched_red"


def test_stick_touching_the_white_ball_is_fine():
    start()
    w = ball(*t.WHITE0)
    stk = stick(x=w.pos[0] - t.BALL_R - t.STICK_LEN / 2, y=w.pos[1])
    t.score(states(white=w, stk=stk))
    assert t._memory["foul"] is None


def test_a_new_trial_clears_a_foul():
    t.score(states(step=40, arms=None))
    assert t._memory["foul"]
    assert t.score(states(step=11, red=ball(*IN_GOAL))) == 1.0


def test_stick_ball_gap():
    s = stick(x=0.0, y=0.0)
    b = ball(t.STICK_LEN / 2 + t.BALL_R + 0.01, 0.0, z=s.pos[2])
    assert t.stick_ball_gap(s, b) == pytest.approx(0.01)


def test_layout():
    assert t.WHITE0[0] + t.BALL_R < t.LINE_X - t.LINE_W / 2  # the white ball starts behind the line
    assert t.RED0[0] - t.BALL_R > t.LINE_X + t.LINE_W / 2  # the red ball starts past it
    assert t.STICK0[0] + t.STICK_LEN / 2 < t.LINE_X - t.LINE_W / 2
    assert t.GOAL_X[0] > t.LINE_X
    assert t.GOAL_WIDTH > 2 * t.BALL_R


def test_task_json_matches_this_file():
    path = Path(t.__file__).parent / "task.json"
    if not path.exists():
        pytest.skip("no task.json yet")
    task = json.loads(path.read_text())
    assert task["instruction"] == t.INSTRUCTION


def test_every_mode_is_a_playable_plan(monkeypatch):
    for mode in ["full", "aim_off", *t.SHORTCUTS]:
        monkeypatch.setenv("DEMO_MODE", mode)
        plan = t.demo()
        assert plan and set(plan) <= {"left", "right"}
        assert all(w.duration > 0 for ws in plan.values() for w in ws)
