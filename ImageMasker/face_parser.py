"""Local BiSeNet semantic face parsing for pixel-accurate privacy masks.

The model architecture is adapted from yakhyo/face-parsing, distributed
under the MIT License. The bundled ResNet-18 weights are trained on
CelebAMask-HQ's 19 semantic face classes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional
from torchvision.models import resnet18


FACE_PARSING_CLASSES = {
    0: "background",
    1: "skin",
    2: "left eyebrow",
    3: "right eyebrow",
    4: "left eye",
    5: "right eye",
    6: "glasses",
    7: "left ear",
    8: "right ear",
    9: "earring",
    10: "nose",
    11: "mouth",
    12: "upper lip",
    13: "lower lip",
    14: "neck",
    15: "necklace",
    16: "clothing",
    17: "hair",
    18: "hat",
}

# Face, facial features, ears, hair, and hats are private. Neck, necklace,
# and clothing remain outside the face-only selection.
PRIVATE_FACE_CLASS_IDS = frozenset(
    {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        17,
        18,
    }
)


class ResNet18FeatureBackbone(nn.Module):
    """ResNet-18 exposing the 1/8, 1/16, and 1/32 feature maps."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # Retained so the module names exactly match the published checkpoint.
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        image = self.conv1(image)
        image = self.bn1(image)
        image = self.relu(image)
        image = self.maxpool(image)
        image = self.layer1(image)
        feature_8 = self.layer2(image)
        feature_16 = self.layer3(feature_8)
        feature_32 = self.layer4(feature_16)
        return feature_8, feature_16, feature_32


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.relu(self.norm(self.conv(features)))


class BiSeNetOutput(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, mid_channels)
        self.conv = nn.Conv2d(
            mid_channels,
            num_classes,
            kernel_size=1,
            bias=False,
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.conv(self.conv_block(features))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, out_channels)
        self.attention = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        refined = self.conv_block(features)
        pooled = functional.adaptive_avg_pool2d(refined, 1)
        return refined * self.attention(pooled)


class ContextPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet18FeatureBackbone()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        feature_8, feature_16, feature_32 = self.backbone(image)
        height_8, width_8 = feature_8.shape[2:]
        height_16, width_16 = feature_16.shape[2:]
        height_32, width_32 = feature_32.shape[2:]

        average = functional.adaptive_avg_pool2d(feature_32, 1)
        average = self.conv_avg(average)
        average = functional.interpolate(
            average,
            (height_32, width_32),
            mode="nearest",
        )

        feature_32 = self.arm32(feature_32) + average
        feature_32_up = functional.interpolate(
            feature_32,
            (height_16, width_16),
            mode="nearest",
        )
        feature_32_up = self.conv_head32(feature_32_up)

        feature_16 = self.arm16(feature_16) + feature_32_up
        feature_16_up = functional.interpolate(
            feature_16,
            (height_8, width_8),
            mode="nearest",
        )
        feature_16_up = self.conv_head16(feature_16_up)
        return feature_8, feature_16_up, feature_32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.conv1 = nn.Conv2d(
            out_channels,
            out_channels // 4,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels // 4,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial: Tensor, context: Tensor) -> Tensor:
        fused = self.conv_block(torch.cat([spatial, context], dim=1))
        attention = functional.adaptive_avg_pool2d(fused, 1)
        attention = self.relu(self.conv1(attention))
        attention = self.sigmoid(self.conv2(attention))
        return fused * attention + fused


class BiSeNet(nn.Module):
    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()
        self.fpn = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, num_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, num_classes)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        height, width = image.shape[2:]
        feature_8, context_8, context_16 = self.fpn(image)
        fused = self.ffm(feature_8, context_8)
        main = functional.interpolate(
            self.conv_out(fused),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        auxiliary_16 = functional.interpolate(
            self.conv_out16(context_8),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        auxiliary_32 = functional.interpolate(
            self.conv_out32(context_16),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        return main, auxiliary_16, auxiliary_32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_privacy_mask(labels: np.ndarray) -> np.ndarray:
    """Convert 19-class face parsing output into a clean binary head mask."""
    if labels.ndim != 2:
        raise ValueError("Face parsing labels must be a two-dimensional array.")

    mask = np.isin(labels, tuple(PRIVATE_FACE_CLASS_IDS)).astype(np.uint8)
    if not np.any(mask):
        return mask

    short_side = max(1, min(mask.shape))
    kernel_size = max(3, int(round(short_side * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 9)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, components, statistics, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if count <= 1:
        return mask * 255

    largest_area = int(statistics[1:, cv2.CC_STAT_AREA].max())
    minimum_area = max(16, int(round(largest_area * 0.01)))
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for component in range(1, count):
        if int(statistics[component, cv2.CC_STAT_AREA]) >= minimum_area:
            cleaned[components == component] = 255

    # Fill enclosed gaps between semantic face parts while preserving the
    # learned outer silhouette around the face and hair.
    contours, _hierarchy = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros_like(cleaned)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def largest_mask_component(mask: np.ndarray) -> np.ndarray:
    """Keep the dominant connected semantic component."""
    binary = (mask > 0).astype(np.uint8)
    count, components, statistics, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        return binary * 255
    largest = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    return (components == largest).astype(np.uint8) * 255


def expanded_face_crop(
    bbox_xyxy: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Expand a localization box to include the whole head and its hair."""
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    side = max(box_width * 1.8, box_height * 1.85)
    center_x = (x1 + x2) / 2.0
    center_y = y1 + box_height * 0.42

    crop_x1 = max(0, int(round(center_x - side / 2.0)))
    crop_y1 = max(0, int(round(center_y - side * 0.55)))
    crop_x2 = min(image_width, int(round(center_x + side / 2.0)))
    crop_y2 = min(image_height, int(round(center_y + side * 0.45)))
    return crop_x1, crop_y1, crop_x2, crop_y2


class SemanticFaceParser:
    """CPU-local BiSeNet inference wrapper."""

    def __init__(
        self,
        model_path: Path,
        *,
        expected_sha256: str,
        input_size: int = 512,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                "The local semantic face parsing model is missing."
            )
        if _sha256(model_path) != expected_sha256.lower():
            raise ValueError(
                "The semantic face parsing model failed its integrity check."
            )

        self.input_size = input_size
        self.model = BiSeNet(num_classes=19)
        try:
            state = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def parse_labels(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError("Face parser input must be a BGR color image.")
        if not crop_bgr.size:
            return np.zeros(crop_bgr.shape[:2], dtype=np.uint8)

        resized = cv2.resize(
            crop_bgr,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).float()
        tensor = tensor.div_(255.0)
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        standard_deviation = tensor.new_tensor(
            (0.229, 0.224, 0.225)
        ).view(3, 1, 1)
        tensor = tensor.sub_(mean).div_(standard_deviation).unsqueeze(0)

        with torch.inference_mode():
            logits = self.model(tensor)[0]
            labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        return cv2.resize(
            labels.astype(np.uint8),
            (crop_bgr.shape[1], crop_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def segment(
        self,
        image_bgr: np.ndarray,
        face_bbox_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        crop_bbox = expanded_face_crop(
            face_bbox_xyxy,
            image_bgr.shape,
        )
        return self.segment_region(image_bgr, crop_bbox)

    def segment_region(
        self,
        image_bgr: np.ndarray,
        crop_bbox_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox_xyxy
        output = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return output

        crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        labels = self.parse_labels(crop)
        output[crop_y1:crop_y2, crop_x1:crop_x2] = (
            semantic_privacy_mask(labels)
        )
        return output
