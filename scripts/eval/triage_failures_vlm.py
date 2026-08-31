"""
VLM-assisted failure triage: what do the failing frames have in common?

THE PROBLEM WITH A RANKED LIST
------------------------------
`prioritize_tests.py` tells you frame 155 of scene-0655 degrades most, and
`fuzz_scenarios.py` tells you occlusion at severity 0.95 on three left-side
cameras is the worst scenario. Neither tells you WHY, and a human has to open
images one at a time to find out. That does not scale past a few dozen frames,
which is exactly why root-cause analysis is the bottleneck in AV validation
rather than failure discovery.

WHAT THIS DOES
--------------
Captions the actual camera images with BLIP (a real VLM: ViT encoder,
cross-attention into a language decoder), then CONTRASTS the caption vocabulary
of the most-sensitive frames against the least-sensitive ones.

The contrast is the whole point. Captioning failures alone produces a list of
sentences about roads and cars, because every frame is a road with cars. What
is diagnostic is which words appear in failing frames *and not* in passing ones.
Scoring is a smoothed log-odds ratio:

    llr(term) = log( (a + alpha) / (n_fail + alpha*V) ) - log( (b + alpha) / (n_pass + alpha*V) )

with add-alpha smoothing so a term appearing 3 times in failures and 0 times in
passes does not produce an infinite score.

HONEST LIMITS
-------------
BLIP was trained on web image-caption pairs, not driving scenes. It will say
"a city street with cars" far more often than "an unprotected left turn with an
occluding bus". Treat the output as a hypothesis generator that tells a human
which frames to open first, not as a labeller. With 82 held-out frames the term
counts are small; the ranking is suggestive, not significant.

Usage
-----
    python scripts/eval/triage_failures_vlm.py \
        --prioritisation outputs/artifacts/test_prioritisation.json \
        --top_k 15 --cams 0,1 \
        --out outputs/artifacts/failure_triage_vlm.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "src")

from opendrivefm.data.nuscenes_mini import NuScenesMiniMultiView  # noqa: E402

CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# Words that appear in nearly every driving caption carry no diagnostic signal.
STOP = set("""a an the of on in at to and or is are with for from by as it its this that
there here be been being was were car cars road street city view image picture photo
shot taken side down up out over near next big small very some many few one two
driving drives driven vehicle vehicles""".split())


def tokenise(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 2 and w not in STOP]


def log_odds(fail: Counter, passc: Counter, alpha: float = 0.5) -> list[tuple[str, float, int, int]]:
    """Smoothed log-odds ratio, failing group versus passing group."""
    vocab = set(fail) | set(passc)
    v, nf, np_ = len(vocab), sum(fail.values()), sum(passc.values())
    out = []
    for w in vocab:
        a, b = fail[w], passc[w]
        lo = (math.log((a + alpha) / (nf + alpha * v))
              - math.log((b + alpha) / (np_ + alpha * v)))
        out.append((w, lo, a, b))
    return sorted(out, key=lambda t: -t[1])


def to_pil(t: torch.Tensor) -> Image.Image:
    """(C,H,W) float tensor -> PIL image."""
    a = t.detach().float().cpu().numpy()
    if a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8).transpose(1, 2, 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prioritisation",
                    default="outputs/artifacts/test_prioritisation.json")
    ap.add_argument("--manifest", default="outputs/artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--label_root", default="outputs/artifacts/nuscenes_labels_128")
    ap.add_argument("--image_hw", default="90,160")
    ap.add_argument("--cams", default="0,1",
                    help="Cameras to caption. Defaults to FRONT and FRONT_LEFT, "
                         "since the fuzzer found left-side cameras dominate the tail.")
    ap.add_argument("--top_k", type=int, default=15,
                    help="Frames in each group (most and least sensitive).")
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--out", default="outputs/artifacts/failure_triage_vlm.json")
    args = ap.parse_args()

    H, W = [int(v) for v in args.image_hw.split(",")]
    cams = [int(c) for c in args.cams.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    prio = json.loads(Path(args.prioritisation).read_text())
    frames = prio["per_frame"]                        # already sorted, most sensitive first
    if len(frames) < 2 * args.top_k:
        raise SystemExit(f"Need at least {2*args.top_k} frames, have {len(frames)}.")
    worst, best = frames[:args.top_k], frames[-args.top_k:]
    print(f"Contrasting {len(worst)} most-sensitive against {len(best)} least-sensitive frames")
    print(f"  most sensitive  max dIoU {worst[0]['sensitivity_max_diou']:+.4f} .. "
          f"{worst[-1]['sensitivity_max_diou']:+.4f}")
    print(f"  least sensitive max dIoU {best[0]['sensitivity_max_diou']:+.4f} .. "
          f"{best[-1]['sensitivity_max_diou']:+.4f}\n")

    from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: E402
    print(f"Loading {args.model}")
    proc = BlipProcessor.from_pretrained(args.model)
    blip = BlipForConditionalGeneration.from_pretrained(args.model).to(device).eval()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    ds = NuScenesMiniMultiView(args.manifest, image_hw=(H, W), frames=1,
                               label_root=args.label_root, return_motion=True,
                               return_trel=True, return_calib=True, augment=False)

    @torch.no_grad()
    def caption_group(group, tag):
        caps, counts = [], Counter()
        for i, rec in enumerate(group):
            mi = rec["manifest_index"]
            x = ds[mi][0]                              # (V, T, C, H, W)
            for c in cams:
                img = to_pil(x[c, -1])
                inp = proc(img, return_tensors="pt").to(device)
                txt = proc.decode(blip.generate(**inp, max_new_tokens=28)[0],
                                  skip_special_tokens=True).strip()
                caps.append({"manifest_index": mi, "scene": rec["scene"],
                             "camera": CAM_NAMES[c],
                             "sensitivity": rec["sensitivity_max_diou"],
                             "worst_fault": rec["worst_fault"], "caption": txt})
                counts.update(set(tokenise(txt)))      # per-image presence, not raw frequency
            if (i + 1) % 5 == 0:
                print(f"  {tag}: {i+1}/{len(group)} frames captioned")
        return caps, counts

    worst_caps, worst_counts = caption_group(worst, "failing")
    best_caps, best_counts = caption_group(best, "passing")

    ranked = log_odds(worst_counts, best_counts)
    print(f"\nDISCRIMINATIVE TERMS (failing frames vs passing frames)")
    print(f"{'term':<18}{'log-odds':>10}{'fail':>7}{'pass':>7}")
    print("-" * 44)
    for w, lo, a, b in ranked[:12]:
        print(f"{w:<18}{lo:>+10.3f}{a:>7}{b:>7}")
    print("\n  ... terms characteristic of the PASSING frames:")
    for w, lo, a, b in ranked[-6:]:
        print(f"{w:<18}{lo:>+10.3f}{a:>7}{b:>7}")

    print(f"\nSAMPLE CAPTIONS, most-sensitive frames")
    for c in worst_caps[:6]:
        print(f"  [{c['scene']} #{c['manifest_index']} {c['camera']} "
              f"dIoU {c['sensitivity']:+.4f} worst={c['worst_fault']}]")
        print(f"    \"{c['caption']}\"")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "vlm": args.model, "cameras": cams, "top_k": args.top_k,
        "n_captions": len(worst_caps) + len(best_caps),
        "discriminative_terms": [
            {"term": w, "log_odds": lo, "n_failing_images": a, "n_passing_images": b}
            for w, lo, a, b in ranked[:40]],
        "failing_captions": worst_caps, "passing_captions": best_caps,
        "caveat": ("BLIP is trained on web image-caption pairs, not driving scenes. "
                   "Output is a hypothesis generator for which frames a human should "
                   "open first, not a labeller. Term counts are small at this dataset "
                   "size; the ranking is suggestive, not significant."),
    }, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
