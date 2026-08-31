"""
Export OpenDriveFM to TorchScript for the C++ runtime, and dump a reference
bundle so C++ output can be checked against Python numerically.

WHAT WAS BROKEN
---------------
The previous version of this script could not produce a loadable graph:

  * it traced with `imgs` shaped (B*V*T, 3, H, W), but OpenDriveFM.forward
    expects (B, V, T, C, H, W);
  * it constructed `OpenDriveFM()` with the default bev_h, which the v11
    architecture asserts against (`assert bev_h == 128`);
  * it traced the raw model, whose forward returns a 4-tuple ending in None.
    A None inside a traced output makes the C++ side awkward and fragile.

WHY THE REFERENCE BUNDLE MATTERS
--------------------------------
Models are trained in Python and shipped in C++. Any divergence between the two
graphs is silent: the C++ binary keeps producing plausible-looking occupancy
grids that no longer match the model that was validated. Every AV programme has
been bitten by this. `odfm_parity_check` loads this bundle, runs the same inputs
through the TorchScript graph in C++, and fails the build if any output drifts
beyond tolerance.

Usage
-----
    python scripts/export_torchscript.py \
        --ckpt outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt \
        --out  outputs/artifacts/opendrivefm_v11.pt \
        --reference outputs/artifacts/parity_reference.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opendrivefm.models.model import OpenDriveFM  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Returns exactly three tensors so the C++ side sees a clean tuple.

    The raw model returns (occ, traj, trust, None); a None element inside a
    traced output is legal but forces awkward IValue handling in C++.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, velocity: torch.Tensor):
        occ, traj, trust, _ = self.model(x, velocity=velocity)
        return occ, traj, trust


class ReferenceBundle(torch.nn.Module):
    """A container of buffers, saved with torch.jit.save.

    C++ reads these by name via `module.attr("input_x").toTensor()`. Using a
    ScriptModule rather than a hand-rolled binary format means the two sides
    cannot disagree about dtype, shape or endianness.
    """

    def __init__(self, x, vel, occ, traj, trust):
        super().__init__()
        self.register_buffer("input_x", x)
        self.register_buffer("input_velocity", vel)
        self.register_buffer("ref_occupancy", occ)
        self.register_buffer("ref_trajectory", traj)
        self.register_buffer("ref_trust", trust)

    def forward(self):
        return (self.input_x, self.input_velocity,
                self.ref_occupancy, self.ref_trajectory, self.ref_trust)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="outputs/artifacts/checkpoints_v11_trustfix2/trust_fixed_v2.ckpt")
    ap.add_argument("--out", default="outputs/artifacts/opendrivefm_v11.pt")
    ap.add_argument("--reference", default="outputs/artifacts/parity_reference.pt")
    ap.add_argument("--bev", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seeds the reference input so the bundle is reproducible.")
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    torch.manual_seed(args.seed)

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt.get("state_dict", ckpt).items()}

    model = OpenDriveFM(bev_h=args.bev, bev_w=args.bev)
    result = model.load_state_dict(state, strict=False)
    n_exp = len(model.state_dict())
    matched = n_exp - len(result.missing_keys)
    print(f"  weights matched: {matched}/{n_exp} ({matched / n_exp:.1%})")
    if matched < n_exp:
        raise SystemExit(
            "Refusing to export a partially loaded model: the exported graph "
            "would not be the model that was validated.")
    model.eval()

    wrapper = ExportWrapper(model).eval()

    # Correct input shape: (B, V, T, C, H, W). The old script passed
    # (B*V*T, C, H, W), which cannot trace against this forward.
    x = torch.randn(args.batch, args.views, args.frames, 3, H, W)
    vel = torch.randn(args.batch, 2)

    print(f"Tracing with x={tuple(x.shape)} velocity={tuple(vel.shape)}")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (x, vel), strict=False)
        traced = torch.jit.freeze(traced)   # inline params: what actually ships

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.out)
    print(f"Saved TorchScript graph: {args.out}")

    # Reference outputs come from the EAGER model, not the traced one. Comparing
    # the traced graph against itself would prove nothing; the point is to catch
    # tracing and freezing from changing behaviour.
    with torch.no_grad():
        occ, traj, trust = wrapper(x, vel)
        t_occ, t_traj, t_trust = traced(x, vel)

    for name, a, b in [("occupancy", occ, t_occ), ("trajectory", traj, t_traj),
                       ("trust", trust, t_trust)]:
        d = (a - b).abs().max().item()
        print(f"  eager vs traced {name:<11} max|delta| = {d:.3e}"
              f"{'  OK' if d < 1e-5 else '  <-- TRACE CHANGED BEHAVIOUR'}")

    bundle = torch.jit.script(ReferenceBundle(x, vel, occ, traj, trust))
    bundle.save(args.reference)
    print(f"Saved parity reference: {args.reference}")
    print(f"  occupancy {tuple(occ.shape)}  trajectory {tuple(traj.shape)}  "
          f"trust {tuple(trust.shape)}")
    print("\nNext:\n"
          "  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \\\n"
          "    -DCMAKE_PREFIX_PATH=\"$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')\"\n"
          "  cmake --build cpp/build -j\n"
          f"  ./cpp/build/odfm_parity_check --model {args.out} --reference {args.reference}")


if __name__ == "__main__":
    main()
