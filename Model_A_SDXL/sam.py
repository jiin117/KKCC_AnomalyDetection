# preprocessing/sam_extractor.py
import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

"""SAM에는 두가지 방식이 있는데 transformers 라이브러리에 있는 SAM을 사용하거나,
 segment_anything 라이브러리에서 제공하는 사용자 편의용 래퍼(Wrapper) 클래스를 사용하는 것이다.
 아래 코드는 segment_anything 라이브러리에서 제공하는 클래스를 사용하는 코드이다."""

"""SAM(Segment Anything Model)을 활용하여 이미지 내 특정 객체의 마스크를 추출하는 모듈"""
class SAMMaskExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. SAM 모델 및 Predictor 세팅
        self.model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(self.device)
        self.predictor = SamPredictor(model)
        print(f"   => [SAMMaskExtractor] SAM 로드 완료 (Device: {self.device})")
        # 2. 기준이 되는 프롬프트 설정 (예: 이미지 중앙 근처의 점 좌표와 라벨)
        # 사진들의 해상도와 대상 위치가 거의 같으므로 동일한 좌표를 재사용합니다.
        self.input_point = np.array([[512, 512]])
        self.input_label = np.array([1]) # 1: 포인트를 포함한 마스크 생성 // 2: 포인트를 제외한 마스크 생성
        """
        for path in image_paths:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 이미지를 세팅하고 프롬프트 주입 (set_image가 매번 호출되지만 좌표 연산은 초고속)
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            # masks[0] 에 저장된 불리언 마스크 배열을 저장하거나 후처리 진행"""

    def extract_mask_by_point(self, pil_image):
        """
        특정 픽셀 좌표(Point)를 기반으로 객체 마스크를 추출합니다.
        point_coords: [[x, y]] 형태의 리스트 (예: 제품의 중심점)
        """
        # 2. PIL 이미지를 SAM이 인식할 수 있는 NumPy 배열(RGB)로 변환
        image_np = np.array(pil_image.convert("RGB"))
        self.predictor.set_image(image_np)

        # 3. SAM 인퍼런스 수행
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False  # 가장 신뢰도 높은 단일 마스크만 반환
        )
        """
        저장 포맷: 추출된 마스크는 cv2.imwrite()를 이용해 흑백(0과 255) .png 파일로 저장해야 
        손실 없이 나중에 StableDiffusionXLInpaintPipeline의 mask_image 인풋으로 바로 활용할 수 있습니다.
        """
        # 4. 부울(Bool) 배열 마스크를 0~255 범위의 PIL 흑백(L 모드) 이미지로 변환
        mask_np = (masks[0] * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np).convert("L")

        return mask_pil