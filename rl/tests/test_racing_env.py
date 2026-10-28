import numpy as np
import pytest

from rl.wrappers import racing_env


class FakeGymEnv:
    def __init__(self, track):
        self.track = track
        self.scan = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.pose = (1.0, 2.0, 0.0)
        self.speed = 0.0
        self.collisions = 0
        self.laps = 0
        self.progress_values = []
        self._progress_index = 0
        self._steps = 0

    def reset(self):
        self._progress_index = 0
        self._steps = 0
        self.collisions = 0

    def drive(self, steer, speed):
        self.last_action = (steer, speed)

    def step(self):
        self._steps += 1
        if self.progress_values:
            self._progress_index = min(
                self._progress_index + 1,
                len(self.progress_values) - 1,
            )
        return True

    def progress(self):
        if not self.progress_values:
            return 0.0
        return self.progress_values[self._progress_index]

    def now(self):
        return self._steps * racing_env.DT

    def close(self):
        pass


@pytest.fixture
def fake_env(monkeypatch):
    instances = []

    def make_env(track):
        env = FakeGymEnv(track)
        instances.append(env)
        return env

    monkeypatch.setattr(racing_env, "GymEnv", make_env)
    wrapped = racing_env.RacingEnv()
    yield wrapped, instances[0]
    wrapped.close()


def test_reward_pays_for_new_progress_not_accumulated_progress(fake_env):
    wrapped, sim = fake_env
    sim.progress_values = [0.0, 0.1, 0.1, 0.15]
    wrapped.reset()

    _, first_reward, _, _, first_info = wrapped.step(np.array([0.0, 0.0]))
    _, repeated_reward, _, _, repeated_info = wrapped.step(np.array([0.0, 0.0]))
    _, later_reward, _, _, later_info = wrapped.step(np.array([0.0, 0.0]))

    assert first_reward == pytest.approx(10.0)
    assert first_info["progress_delta"] == pytest.approx(0.1)
    assert repeated_reward == pytest.approx(0.0)
    assert repeated_info["progress_delta"] == pytest.approx(0.0)
    assert later_reward == pytest.approx(5.0)
    assert later_info["progress_reward"] == pytest.approx(5.0)


def test_reward_does_not_pay_again_for_progress_when_progress_regresses(fake_env):
    wrapped, sim = fake_env
    sim.progress_values = [0.0, 0.2, 0.1]
    wrapped.reset()

    _, forward_reward, _, _, _ = wrapped.step(np.array([0.0, 0.0]))
    _, backward_reward, _, _, info = wrapped.step(np.array([0.0, 0.0]))

    assert forward_reward == pytest.approx(20.0)
    assert backward_reward == pytest.approx(0.0)
    assert info["progress_delta"] == pytest.approx(0.0)


def test_collision_penalty_is_applied_to_progress_reward(fake_env):
    wrapped, sim = fake_env
    sim.progress_values = [0.0, 0.1]
    wrapped.reset()
    sim.collisions = 1

    _, reward, terminated, truncated, info = wrapped.step(np.array([0.0, 0.0]))

    assert reward == pytest.approx(-40.0)
    assert terminated
    assert not truncated
    assert info["progress_reward"] == pytest.approx(10.0)


def test_reset_clears_progress_reward_baseline(fake_env):
    wrapped, sim = fake_env
    sim.progress_values = [0.0, 0.1]
    wrapped.reset()
    wrapped.step(np.array([0.0, 0.0]))

    sim.progress_values = [0.0, 0.1]
    _, info = wrapped.reset()
    _, reward, _, _, step_info = wrapped.step(np.array([0.0, 0.0]))

    assert info == {}
    assert reward == pytest.approx(10.0)
    assert step_info["progress_delta"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (np.array([0.0]), "shape"),
        (np.array([0.0, 0.0, 0.0]), "shape"),
        (np.array([np.nan, 0.0]), "finite"),
        (np.array([0.0, np.inf]), "finite"),
    ],
)
def test_rejects_malformed_actions(fake_env, action, message):
    wrapped, _ = fake_env

    with pytest.raises(ValueError, match=message):
        wrapped._scale(action)


def test_action_scaling_clips_valid_actions_to_physical_limits(fake_env):
    wrapped, _ = fake_env

    steer, speed = wrapped._scale(np.array([-2.0, 2.0]))

    assert steer == pytest.approx(-racing_env.MAX_STEER)
    assert speed == pytest.approx(racing_env.MAX_SPEED)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": 1.5}, "max_steps"),
        ({"target_laps": 0}, "target_laps"),
        ({"target_laps": True}, "target_laps"),
    ],
)
def test_rejects_invalid_episode_limits(monkeypatch, kwargs, message):
    def unexpected_env_creation(*args, **kwargs):
        pytest.fail("invalid options should be rejected before creating GymEnv")

    monkeypatch.setattr(racing_env, "GymEnv", unexpected_env_creation)

    with pytest.raises(ValueError, match=message):
        racing_env.RacingEnv(**kwargs)
