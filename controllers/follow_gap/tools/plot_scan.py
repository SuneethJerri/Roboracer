"""Polar plots of what FTG saw and chose, for chosen steps of a saved run (D1, D4).

Left panel : the scan (grey = raw, blue = processed), search window, safety bubble
             (red), every gap in its own colour (chosen gap outlined), heading arrow.
Right panel: the car on the map (if --map is given) and the lap trajectory.

Examples (repo root, f1env active):
  python controllers\\follow_gap\\tools\\plot_scan.py --run controllers\\follow_gap\\data\\runs\\baseline_001.npz --steps 50 120 300
  python controllers\\follow_gap\\tools\\plot_scan.py --run <run.npz> --steps 120 --variant repaired --map C:\\Users\\abhis\\f1tenth_gym\\examples\\example_map.yaml
  python controllers\\follow_gap\\tools\\plot_scan.py --run <run.npz> --every 40     # one plot per 40 steps

The controller is re-run on the saved scan, so you can plot a run recorded with
the baseline using the repaired controller (--variant repaired) and compare.
Steering smoothing depends on history and is not replayed; gap choice does not.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

HERE = Path(__file__).resolve()
FG_DIR = HERE.parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(FG_DIR))

COLORS = ["tab:green", "tab:orange", "tab:purple", "tab:cyan", "tab:olive", "tab:pink"]


def load_controller(variant, cfg):
    if variant == "repaired":
        from ftg_repaired import FollowGapRepaired
        return FollowGapRepaired(cfg)
    from ftg import FollowGap
    return FollowGap(cfg)


def load_map(map_yaml):
    import matplotlib.pyplot as plt
    y = yaml.safe_load(open(map_yaml))
    img = plt.imread(Path(map_yaml).parent / y["image"])
    res = float(y["resolution"])
    ox, oy = float(y["origin"][0]), float(y["origin"][1])
    h, w = img.shape[:2]
    return img, [ox, ox + w * res, oy, oy + h * res]


def make_figure(ctrl, scan, d, traj, pose, title, map_info=None):
    import matplotlib.pyplot as plt
    ang, R = ctrl.angles, ctrl.cfg["max_range"]
    fig = plt.figure(figsize=(13, 6.2))

    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)                     # counter-clockwise = left = positive
    try:
        ax.set_thetamin(-135)
        ax.set_thetamax(135)
    except Exception:
        pass
    ax.set_ylim(0, R)
    ax.plot(ang, np.clip(np.nan_to_num(scan, nan=R, posinf=R), 0, R),
            color="0.55", lw=0.7, label="raw scan")
    ax.fill_between(ang, 0, d["proc"], color="tab:blue", alpha=0.25, label="processed scan")
    for edge in (ctrl.lo, ctrl.hi - 1):
        ax.plot([ang[edge]] * 2, [0, R], "k--", lw=0.8)
    b0 = max(0, d["bubble_center"] - d["bubble_half"])
    b1 = min(len(ang), d["bubble_center"] + d["bubble_half"] + 1)
    ax.fill_between(ang[b0:b1], 0, R, color="red", alpha=0.35, label="safety bubble")
    for i, (s, e) in enumerate(d["gaps"]):
        ax.fill_between(ang[s:e], 0, R, color=COLORS[i % len(COLORS)], alpha=0.18,
                        label=f"gap {i + 1} ({e - s} beams)")
    if d["chosen_gap"]:
        s, e = d["chosen_gap"]
        ax.plot(ang[s:e], np.full(e - s, R * 0.98), color="k", lw=3, label="chosen gap")
    ax.annotate("", xy=(d["heading"], R * 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5))
    ax.legend(loc="lower left", bbox_to_anchor=(-0.08, -0.16), fontsize=7)
    ax.set_title(title, fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    if map_info is not None:
        ax2.imshow(map_info[0], extent=map_info[1], cmap="gray", origin="upper")
    ax2.plot(traj[:, 0], traj[:, 1], color="tab:blue", lw=1, alpha=0.6, label="trajectory")
    x, y, th = pose
    ax2.plot(x, y, "o", color="crimson", ms=7, label="car")
    ax2.annotate("", xy=(x + 0.8 * np.cos(th), y + 0.8 * np.sin(th)), xytext=(x, y),
                 arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2))
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", required=True, help="run .npz from record_run.py")
    p.add_argument("--steps", nargs="*", type=int, default=[], help="step indices to plot")
    p.add_argument("--every", type=int, default=0, help="plot every N-th step instead")
    p.add_argument("--variant", choices=["baseline", "repaired"], default="baseline")
    p.add_argument("--ctrl_config", default=None)
    p.add_argument("--map", default=None, help="map .yaml (e.g. example_map.yaml) for the right panel")
    p.add_argument("--out", default=str(FG_DIR / "data" / "plots"))
    p.add_argument("--show", action="store_true", help="open windows instead of only saving")
    a = p.parse_args()
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(a.run)
    n = len(d["t"])
    if a.steps:
        steps = list(a.steps)
    elif a.every > 0:
        steps = list(range(0, n, a.every))
    else:
        steps = [n // 2]                          # no choice given: plot the middle step
    default_cfg = "follow_gap_repaired.yaml" if a.variant == "repaired" else "follow_gap.yaml"
    cfg = yaml.safe_load(open(a.ctrl_config or REPO / "configs" / default_cfg))
    ctrl = load_controller(a.variant, cfg)
    map_info = load_map(a.map) if a.map else None
    traj = np.stack([d["x"], d["y"]], axis=1)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in steps:
        if not 0 <= i < n:
            print(f"skip step {i}: run has {n} steps")
            continue
        ctrl.reset()
        steer, speed = ctrl.act(d["scan"][i])
        dbg = ctrl.debug
        title = (f"{Path(a.run).stem} | {a.variant} | step {i} | t={d['t'][i]:.2f}s | "
                 f"heading={np.rad2deg(dbg['heading']):+.1f} deg | "
                 f"front range={dbg['front_range']:.2f} m")
        fig = make_figure(ctrl, d["scan"][i], dbg, traj,
                          (d["x"][i], d["y"][i], d["theta"][i]), title, map_info)
        path = out_dir / f"{Path(a.run).stem}_{a.variant}_step{i:05d}.png"
        fig.savefig(path, dpi=130)
        print("saved", path)
        if a.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
