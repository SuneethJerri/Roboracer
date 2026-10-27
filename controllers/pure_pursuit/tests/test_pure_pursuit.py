import numpy as np
import pytest

from controllers.pure_pursuit.pure_pursuit import (
    PurePursuit,
    curvature_from_path,
    steering_from_local_target,
)


def test_circle_steering_matches_analytic_value():
    wheelbase = 0.33
    radius = 2.0
    phi = np.deg2rad(25.0)
    # Point on a circle of radius R in the vehicle frame.
    x = radius * np.sin(phi)
    y = radius * (1.0 - np.cos(phi))
    got = steering_from_local_target(x, y, wheelbase)
    expected = np.arctan(wheelbase / radius)
    assert got == pytest.approx(expected, abs=1e-10)


def test_straight_target_gives_zero_steering():
    assert steering_from_local_target(2.0, 0.0, 0.33) == pytest.approx(0.0)


def test_target_to_left_gives_positive_steering():
    assert steering_from_local_target(2.0, 0.5, 0.33) > 0


def test_target_to_right_gives_negative_steering():
    assert steering_from_local_target(2.0, -0.5, 0.33) < 0


def test_zero_distance_target_is_safe():
    assert steering_from_local_target(0.0, 0.0, 0.33) == 0.0


def test_curvature_estimate_on_circle():
    radius = 3.0
    angles = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    k = curvature_from_path(points)
    assert np.median(np.abs(k)) == pytest.approx(1.0 / radius, rel=0.02)


def test_speed_decreases_as_curvature_increases():
    ctrl = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 5.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
    })
    assert ctrl.speed_from_curvature(0.5) < ctrl.speed_from_curvature(0.05)
    assert ctrl.speed_from_curvature(0.0) == pytest.approx(5.0)


def test_adaptive_lookahead_increases_with_speed():
    ctrl = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 5.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
    })
    assert ctrl.lookahead_distance(4.0) > ctrl.lookahead_distance(1.0)


def test_controller_requires_path_before_acting():
    ctrl = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 2.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
    })
    with pytest.raises(RuntimeError, match="no centreline loaded"):
        ctrl.act(np.zeros(10), (0.0, 0.0, 0.0))


def test_controller_tracks_a_closed_circle_at_circle_start():
    ctrl = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 2.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
    })
    radius = 3.0
    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    ctrl.set_path(points)
    # At the rightmost point of this CCW circle, tangent points up (+y), heading pi/2.
    steer, speed = ctrl.act(None, (radius, 0.0, np.pi / 2))
    assert steer > 0
    assert 0.0 < speed <= 2.0
