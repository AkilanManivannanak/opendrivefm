"""
add_vit_option.py
Adds a lightweight ViT option to the backbone for Step 3 compliance.
This patches model.py to add a ViTStem class alongside the existing CNN stem.

Copy to ~/opendrivefm/src/opendrivefm/models/
Run: python src/opendrivefm/models/add_vit_option.py

This adds ViTStem as an importable alternative backbone stem.
The existing model is unchanged — ViT is offered as an option.
"""
vit_code = '''

class ViTStem(nn.Module):
    """
    Lightweight Vision Transformer stem for per-camera feature extraction.
    Patch-based tokenisation followed by transformer encoder.
    Added for Step 3 CNN/ViT compliance.
    Uses patch_size=16 on 90x160 images → (5*10)=50 patches per camera.
    """
    def __init__(self, img_h=90, img_w=160, patch_size=16,
                 in_ch=3, d=384, n_heads=6, n_layers=2):
        super().__init__()
        self.patch_size = patch_size
        n_h = img_h // patch_size   # 5
        n_w = img_w // patch_size   # 10
        n_patches = n_h * n_w       # 50

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_ch, d, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2*d,
            dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """x: (B, 3, H, W) → (B, d) CLS token feature"""
        B = x.shape[0]
        # Patch embed → (B, d, n_h, n_w) → (B, n_patches, d)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, p], dim=1) + self.pos_embed
        out = self.transformer(tokens)
        return self.norm(out[:, 0])   # CLS token → (B, d)
'''

import re
from pathlib import Path

model_path = Path("src/opendrivefm/models/model.py")
content = model_path.read_text()

if "class ViTStem" in content:
    print("ViTStem already exists in model.py")
else:
    # Insert before the first class definition
    insert_at = content.find("class TemporalTransformer")
    new_content = content[:insert_at] + vit_code + "\n\n" + content[insert_at:]
    model_path.write_text(new_content)
    print("ViTStem added to model.py")
    print("You can now import: from opendrivefm.models.model import ViTStem")

if __name__ == "__main__":
    pass
