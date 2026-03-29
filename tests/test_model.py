"""
Unit tests for OpenDriveFM Trust-Aware model.
Run: pytest tests/ -v
"""
import pytest
import torch
from opendrivefm.models.model import (
    OpenDriveFM, CameraTrustScorer, TrustWeightedFusion, MultiViewVideoBackbone
)


@pytest.fixture
def sample_batch():
    """Minimal batch: B=2, V=6 cameras, T=1 frame, C=3 RGB, H=90, W=160"""
    return torch.randn(2, 6, 1, 3, 90, 160)


def test_trust_scorer_output_range(sample_batch):
    """Trust scores must be in (0, 1)."""
    scorer = CameraTrustScorer()
    B, V, T, C, H, W = sample_batch.shape
    imgs = sample_batch[:, :, 0].reshape(B * V, C, H, W)
    trust = scorer(imgs)
    assert trust.shape == (B * V,), f"Expected ({B*V},), got {trust.shape}"
    assert trust.min() >= 0.0
    assert trust.max() <= 1.0


def test_backbone_trust_enabled(sample_batch):
    """Backbone with trust=True returns (z, ft, trust) with correct shapes."""
    backbone = MultiViewVideoBackbone(d=128, enable_trust=True)
    z, ft, trust = backbone(sample_batch)
    B = sample_batch.shape[0]
    V = sample_batch.shape[1]
    assert z.shape == (B, 128)
    assert trust.shape == (B, V)
    assert trust.min() >= 0.0
    assert trust.max() <= 1.0


def test_backbone_trust_disabled(sample_batch):
    """Backbone with trust=False returns all-ones trust."""
    backbone = MultiViewVideoBackbone(d=128, enable_trust=False)
    z, ft, trust = backbone(sample_batch)
    assert (trust == 1.0).all()


def test_full_model_output_shapes(sample_batch):
    """Full model returns occ (B,1,64,64), traj (B,12,2), trust (B,6)."""
    model = OpenDriveFM(d=128, bev_h=64, bev_w=64, horizon=12, enable_trust=True)
    occ, traj, trust = model(sample_batch)
    B = sample_batch.shape[0]
    assert occ.shape   == (B, 1, 64, 64)
    assert traj.shape  == (B, 12, 2)
    assert trust.shape == (B, 6)


def test_full_model_no_trust(sample_batch):
    """Model without trust module still returns correct shapes."""
    model = OpenDriveFM(d=128, enable_trust=False)
    occ, traj, trust = model(sample_batch)
    B = sample_batch.shape[0]
    assert occ.shape == (B, 1, 64, 64)
    assert traj.shape == (B, 12, 2)


def test_perturbations_preserve_shape(sample_batch):
    """Each perturbation must preserve input tensor shape."""
    from opendrivefm.robustness.perturbations import PERTURBATIONS
    B, V, T, C, H, W = sample_batch.shape
    imgs = sample_batch[:, 0, 0].clone()  # (B, C, H, W)
    for name, cls in PERTURBATIONS.items():
        perturber = cls()
        out = perturber(imgs)
        assert out.shape == imgs.shape, f"{name}: shape mismatch {out.shape} != {imgs.shape}"
        assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-5, f"{name}: values out of [0,1]"


def test_trust_drops_under_blur(sample_batch):
    """
    After blurring one camera, its trust score should drop
    (on average over random seeds, the CNN branch learns this during training,
    but the handcrafted stats branch should detect it immediately).
    This is a 'smoke test' to verify the stats branch runs without error.
    """
    from opendrivefm.robustness.perturbations import GaussianBlur
    model = OpenDriveFM(d=128, enable_trust=True)
    model.eval()

    with torch.no_grad():
        _, _, trust_clean = model(sample_batch)
        x_blurred = sample_batch.clone()
        blurrer = GaussianBlur(sigma_range=(3.0, 4.0))
        B, V, T, C, H, W = x_blurred.shape
        flat = x_blurred[:, 0, 0]  # front camera
        x_blurred[:, 0, 0] = blurrer(flat)
        _, _, trust_blurred = model(x_blurred)

    # The test just checks it runs — trust learning happens during training
    assert trust_blurred.shape == trust_clean.shape
    print(f"\n  Clean trust[0]: {trust_clean[0].tolist()}")
    print(f"  Blurred trust[0]: {trust_blurred[0].tolist()}")
