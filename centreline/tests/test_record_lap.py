import numpy as np

from centreline.tools.record_lap import COLUMNS, load, record, save


class FakeEnv:
    """Drives round a 5 m circle at 1 m/s; finishes after one lap."""

    def __init__(self):
        self._steps = 0

    def now(self):
        return self._steps * 0.01

    @property
    def laps(self):
        return int(self.now() / (2 * np.pi * 5.0))

    @property
    def pose(self):
        a = self.now() / 5.0
        return 5.0 * np.cos(a), 5.0 * np.sin(a), a + np.pi / 2

    speed = 1.0
    scan = None
    collisions = 0

    def progress(self):
        return self.now() / (2 * np.pi * 5.0)

    def drive(self, steer, speed):
        pass

    def step(self):
        self._steps += 1
        return True


class FakeController:
    def reset(self):
        pass

    def act(self, scan, pose):
        return 0.0, 1.0


def test_record_adds_noise_and_keeps_truth(tmp_path):
    rows, outcome = record(FakeEnv(), FakeController(), rate_hz=10.0, noise=0.05, seed=1)

    assert outcome["complete"]
    assert rows.shape[1] == len(COLUMNS)
    assert abs(len(rows) - 10 * 2 * np.pi * 5.0) < 3          # 10 Hz over one lap
    truth_radius = np.hypot(rows[:, 5], rows[:, 6])
    assert np.allclose(truth_radius, 5.0)
    error = np.hypot(rows[:, 1] - rows[:, 5], rows[:, 2] - rows[:, 6])
    assert 0.02 < error.mean() < 0.1


def test_csv_round_trip(tmp_path):
    rows, _ = record(FakeEnv(), FakeController(), noise=0.0)
    data = load(save(rows, tmp_path / "lap.csv"))
    assert list(data) == list(COLUMNS)
    assert np.allclose(data["x"], rows[:, 1], atol=1e-6)
