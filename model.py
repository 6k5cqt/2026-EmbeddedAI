"""
model.py - Center Lane Detection Model (Heatmap version)

Output: (2, HEATMAP_BINS)
  [0] center heatmap  - 센터라인 후보 확률 분포
  [1] solid heatmap   - solid 라인 확률 분포
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.signal import find_peaks

HEATMAP_BINS = 64


class CenterModel(nn.Module):
    def __init__(self, backbone='mobilenetv2', pretrained=True):
        super().__init__()

        if backbone == 'resnet18':
            base = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone == 'mobilenetv2':
            base = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            feat_dim = base.classifier[-1].in_features
            base.classifier = nn.Identity()
        elif backbone == 'alexnet':
            base = models.alexnet(
                weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
            feat_dim = 4096
            base.classifier[-1] = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2 * HEATMAP_BINS),  # center + solid
        )

    def forward(self, x):
        feat = self.backbone(x)
        out  = self.head(feat)                          # (B, 2*BINS)
        out  = out.view(-1, 2, HEATMAP_BINS)            # (B, 2, BINS)
        out  = torch.sigmoid(out)                       # [0, 1]
        return out


def get_model(backbone='mobilenetv2', pretrained=True):
    return CenterModel(backbone=backbone, pretrained=pretrained)


def load_model(path, backbone='mobilenetv2'):
    model = get_model(backbone=backbone, pretrained=False)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def decode(output, width, y_px, conf_thr=0.3, max_centers=3):
    """
    output  : numpy (2, BINS) or tensor
    width   : image width in pixels
    y_px    : fixed y pixel

    Returns:
        centers      : list of x_px (센터라인 피크, 최대 max_centers개)
        solid_left   : x_px or None
        solid_right  : x_px or None
        center_final : 최종 사용할 x_px
        y            : y_px
        center_heatmap : (BINS,) numpy - 시각화용
        solid_heatmap  : (BINS,) numpy - 시각화용
    """
    if hasattr(output, 'cpu'):
        output = output.cpu().numpy()

    center_hm = output[0]  # (BINS,)
    solid_hm  = output[1]  # (BINS,)

    def bin_to_x(b):
        return int((b + 0.5) / HEATMAP_BINS * width)

    def get_peaks(hm, thr, max_n):
        peaks, props = find_peaks(hm, height=thr, distance=3)
        if len(peaks) == 0:
            return []
        # 높은 순으로 정렬
        order = np.argsort(props['peak_heights'])[::-1]
        peaks = peaks[order[:max_n]]
        # x 위치 순으로 재정렬
        peaks = sorted(peaks)
        return [bin_to_x(p) for p in peaks]

    centers     = get_peaks(center_hm, conf_thr, max_centers)
    solid_peaks = get_peaks(solid_hm,  conf_thr, 2)

    # solid는 가장 왼쪽/오른쪽 피크
    solid_left  = solid_peaks[0]  if len(solid_peaks) >= 1 else None
    solid_right = solid_peaks[-1] if len(solid_peaks) >= 2 else None

    # 최종 센터 결정
    if centers:
        center_final = centers[0]
    elif solid_left is not None and solid_right is not None:
        center_final = (solid_left + solid_right) // 2
    elif solid_left is not None:
        center_final = solid_left
    elif solid_right is not None:
        center_final = solid_right
    else:
        center_final = width // 2

    return {
        'centers':        centers,
        'solid_left':     solid_left,
        'solid_right':    solid_right,
        'center_final':   center_final,
        'y':              y_px,
        'center_heatmap': center_hm,
        'solid_heatmap':  solid_hm,
    }
