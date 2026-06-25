# preprocessing/inference.py
import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline


class SDXLInpaintGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        print("   => [SDXLInpaintGenerator] 인페인팅 베이스 모델 로드 완료.")

    def generate_synthetic_data(self, lora_path, image_dir, output_gen_dir, prompt, sam_extractor):
        """
        sam_extractor 인스턴스를 받아 실시간으로 마스크를 생성하고 가상 결함을 인페인팅합니다.
        """
        if os.path.exists(lora_path):
            self.pipe.load_lora_weights(lora_path)
            print(f"   => [SDXLInpaintGenerator] {os.path.basename(lora_path)} 결합 완료.")

        os.makedirs(output_gen_dir, exist_ok=True)

        for img_name in os.listdir(image_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                init_image = Image.open(img_path).convert("RGB")

                # [수정 완료] 더미 마스크 대신 SAM 모듈을 호출하여 실제 물리 마스크 획득
                # 1024x1024 해상도 기준, 제품이 위치한 중앙부(512, 512)를 포인트 프롬프트로 지정 예시
                center_point = [[512, 512]]
                mask_image = sam_extractor.extract_mask_by_point(init_image, point_coords=center_point)

                # 4. 가이드 문서 규격 준수 이미지 생성 (cross_attention_kwargs 반영)
                with torch.no_grad():
                    result_image = self.pipe(
                        prompt=prompt,
                        image=init_image,
                        mask_image=mask_image,
                        num_inference_steps=30,
                        cross_attention_kwargs={"scale": 0.8}  # LoRA 가중치 강도 조절
                    ).images[0]

                # 결과 저장
                result_image.save(os.path.join(output_gen_dir, f"gen_{img_name}"))

        self.pipe.unload_lora_weights()