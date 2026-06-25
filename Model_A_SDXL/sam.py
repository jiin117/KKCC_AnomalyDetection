# preprocessing/sam_extractor.py
import os
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


class SAMMaskExtractor:
    """SAM(Segment Anything Model)을 활용하여 이미지 내 특정 객체의 마스크를 추출하는 모듈"""

    def __init__(self, model_type="vit_h", checkpoint_path="./sam_vit_h_4b8939.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM 가중치 파일이 존재하지 않습니다: {checkpoint_path}")

        # 1. SAM 모델 로드 및 Predictor 초기화 (VRAM 48GB 환경 공유)
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print(f"   => [SAMMaskExtractor] SAM ({model_type}) 로드 완료 (Device: {self.device})")

    def extract_mask_by_point(self, pil_image, point_coords, point_labels=[1]):
        """
        특정 픽셀 좌표(Point)를 기반으로 객체 마스크를 추출합니다.
        point_coords: [[x, y]] 형태의 리스트 (예: 제품의 중심점)
        """
        # 2. PIL 이미지를 SAM이 인식할 수 있는 NumPy 배열(RGB)로 변환
        image_np = np.array(pil_image.convert("RGB"))
        self.predictor.set_image(image_np)

        # 3. SAM 인퍼런스 수행
        input_point = np.array(point_coords)
        input_label = np.array(point_labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False  # 가장 신뢰도 높은 단일 마스크만 반환
        )

        # 4. 부울(Bool) 배열 마스크를 0~255 범위의 PIL 흑백(L 모드) 이미지로 변환
        mask_np = (masks[0] * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np).convert("L")

        return mask_pil