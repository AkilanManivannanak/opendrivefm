"""
Release-readiness gates for OpenDriveFM.

Every measurement this repo makes is written to a JSON artifact. This script
turns those artifacts into pass/fail gates so a regression becomes a red build
instead of something discovered months later while writing a README.

The gates deliberately encode what was actually learned, not round numbers:

  detection_min_auroc     Pooled AUROC must beat chance with a confidence
                          interval that excludes 0.5. The v11 baseline measured
                          0.434 with CI [0.419, 0.449] -- significantly INVERTED.
                          A gate on the point estimate alone would have passed a
                          detector that grew more confident as cameras failed.

  per_fault_min_auroc     No single fault may collapse. A pooled-only gate let
                          an earlier fix reach 0.652 overall while blur sat at
                          0.249, because the objective traded the hard fault for
                          the easy ones. Hazard-aligned validation gates the
                          worst case, not the average.

  parity                  The C++ graph must still reproduce the validated
                          Python model. Silent numerical drift between the
                          trained model and the shipped one is the failure mode
                          this whole harness exists to prevent.

  latency_p95_ms          p95, not p99.9. Repeated runs on the same machine
                          showed p50 stable to 1.5% while p99.9 moved 53% with
                          background load. Gating on an unreproducible statistic
                          produces flaky builds and teaches people to ignore CI.

Usage:
    python scripts/ci/check_gates.py --config configs/gates.yaml
    python scripts/ci/check_gates.py --detection outputs/artifacts/ood_detection_report_v11_trustfix2.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

DEFAULTS = {
    "detection_min_auroc": 0.70,
    "detection_ci_must_exclude_chance": True,
    "per_fault_min_auroc": 0.45,     # below this a fault is actively inverted
    "per_fault_target_auroc": 0.80,  # reported, not enforced
    "latency_p95_ms": 35.0,
    "require_determinism": True,
}


class Gates:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append((name, bool(ok), detail))

    def note(self, name: str, detail: str) -> None:
        self.rows.append((name, None, detail))

    def report(self) -> int:
        width = max(len(r[0]) for r in self.rows) + 2
        failed = 0
        for name, ok, detail in self.rows:
            tag = "NOTE" if ok is None else ("PASS" if ok else "FAIL")
            if ok is False:
                failed += 1
            print(f"  [{tag}] {name:<{width}} {detail}")
        print()
        if failed:
            print(f"{failed} gate(s) FAILED. This artifact is not release-ready.")
        else:
            print("All gates passed.")
        return 1 if failed else 0


def gate_detection(g: Gates, path: Path, cfg: dict) -> None:
    d = json.loads(path.read_text())
    overall = d["overall"]
    best_name, best = max(overall.items(), key=lambda kv: kv[1]["auroc"])
    lo, hi = best["auroc_ci95"]

    g.check("detection.auroc", best["auroc"] >= cfg["detection_min_auroc"],
            f"{best['auroc']:.4f} (detector '{best_name}', floor "
            f"{cfg['detection_min_auroc']:.2f})")

    if cfg["detection_ci_must_exclude_chance"]:
        g.check("detection.ci_excludes_chance", lo > 0.5,
                f"95% CI [{lo:.4f}, {hi:.4f}] "
                f"{'excludes' if lo > 0.5 else 'INCLUDES OR SITS BELOW'} 0.5")

    by_fault: dict[str, list[float]] = collections.defaultdict(list)
    for v in d["per_fault"].values():
        by_fault[v["fault"]].append(v["detectors"][best_name]["auroc"])
    means = {k: sum(v) / len(v) for k, v in by_fault.items()}
    worst_fault = min(means, key=means.get)

    g.check("detection.worst_fault", means[worst_fault] >= cfg["per_fault_min_auroc"],
            f"'{worst_fault}' {means[worst_fault]:.4f} (floor "
            f"{cfg['per_fault_min_auroc']:.2f})")

    below = {k: v for k, v in means.items() if v < cfg["per_fault_target_auroc"]}
    if below:
        g.note("detection.below_target",
               ", ".join(f"{k} {v:.3f}" for k, v in sorted(below.items(), key=lambda x: x[1]))
               + f" (target {cfg['per_fault_target_auroc']:.2f})")


def gate_latency(g: Gates, path: Path, cfg: dict) -> None:
    d = json.loads(path.read_text())
    g.check("latency.p95_ms", d["p95_ms"] <= cfg["latency_p95_ms"],
            f"{d['p95_ms']:.3f} ms (budget {cfg['latency_p95_ms']:.1f} ms, "
            f"p50 {d['p50_ms']:.3f})")
    # Reported, never gated: see the module docstring.
    g.note("latency.tail",
           f"p99 {d['p99_ms']:.3f} ms, p99.9 {d['p999_ms']:.3f} ms, "
           f"jitter p99/p50 {d['jitter_p99_over_p50']:.3f} "
           f"(not gated: tail is not reproducible run-to-run)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detection", type=Path,
                    default=Path("outputs/artifacts/ood_detection_report_v11_trustfix2.json"))
    ap.add_argument("--latency", type=Path, default=None,
                    help="JSON emitted by odfm_parity_check (the 'JSON {...}' line).")
    ap.add_argument("--parity-exit-code", type=int, default=None,
                    help="Exit code from odfm_parity_check; 0 means the C++ graph matches.")
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            ap.add_argument(f"--{key.replace('_', '-')}", type=lambda s: s.lower() == "true",
                            default=val)
        else:
            ap.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}

    g = Gates()
    print("OpenDriveFM release gates\n")

    if args.detection and args.detection.exists():
        gate_detection(g, args.detection, cfg)
    else:
        g.check("detection.report", False, f"missing: {args.detection}")

    if args.latency:
        if args.latency.exists():
            gate_latency(g, args.latency, cfg)
        else:
            g.check("latency.report", False, f"missing: {args.latency}")

    if args.parity_exit_code is not None:
        g.check("parity.cpp_matches_python", args.parity_exit_code == 0,
                f"odfm_parity_check exit code {args.parity_exit_code}")

    return g.report()


if __name__ == "__main__":
    sys.exit(main())
