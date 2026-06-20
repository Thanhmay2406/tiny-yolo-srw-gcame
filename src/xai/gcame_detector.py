from __future__ import annotations

from torch import nn


class GCAMEPlaceholder(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise NotImplementedError(
            "G-CAME is not implemented in this repository yet. "
            "Use saliency_head, gt_mask_debug, offline_xai_teacher, or gradcam_like_online_debug."
        )
