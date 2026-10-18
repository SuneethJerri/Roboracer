"""The shared runner imports controllers as packages; make sure that works."""
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_importable_as_package():
    sys.path.insert(0, str(REPO))
    pkg = importlib.import_module("controllers.follow_gap")
    assert hasattr(pkg, "FollowGap")
    assert hasattr(pkg, "FollowGapRepaired")


def test_both_classes_expose_act():
    sys.path.insert(0, str(REPO))
    pkg = importlib.import_module("controllers.follow_gap")
    for cls in (pkg.FollowGap, pkg.FollowGapRepaired):
        assert callable(getattr(cls, "act"))
        assert callable(getattr(cls, "reset"))
