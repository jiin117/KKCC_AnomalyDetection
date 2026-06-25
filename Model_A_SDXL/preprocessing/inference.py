import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline


class SDXLInpaintGenerator:
    """3단계: SAM 마스크와 품목별 LoRA 가중치를 활용하여 증강 이미지를 생성하는 모듈"""
    def __init__(self):
        """
        2단계 학습은 Hugging Face accelerate 라이브러리가
        내부적으로 하드웨어(CUDA) 설정을 자동으로 감지하고 제어해 준 것입니다.
        하지만 3단계 파이프라인(StableDiffusionXLInpaintPipeline)은
        순수 PyTorch와 diffusers 인스턴스 기반으로 동작하므로,
        개발자가 "이 모델은 GPU(cuda)로 보내라"고 명시적으로 지시(.to("cuda"))하지 않으면
        기본값인 CPU에서 모델을 로드하고 연산을 수행합니다."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        print("=> 인페인팅 베이스 모델 로드 완료.")
        self.category = None
        self.lora_dir = None
        self.input_dir = None
        self.output_dir = None
        self.prompt = None

    def generate_image(self, category, lora_dir, input_dir, output_dir, prompt):
        """특정 품목의 LoRA를 로드하여 마스크 기반 인페인팅 이미지를 생성하는 메서드"""
        self.category = category
        self.lora_dir = lora_dir
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, category)
        self.prompt = prompt
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(self.lora_dir):
            self.pipe.load_lora_weights(self.lora_dir)
            print(f" => [SDXLInpaintGenerator] {os.path.basename(self.lora_dir)} 결합 완료.")

        # 3. 디렉토리 내부 원본 이미지들을 순회하며 SAM 마스크와 결합 생성 (루프)
        for img_name in os.listdir(self.input_dir):
            if img_name.lower().endswith('.png'):
                img_path = os.path.join(self.input_dir, img_name)
                init_image = Image.open(img_path).convert("RGB")

                # [참고] SAM 모듈을 통해 mask_image를 추출하는 로직이 여기에 연동되어야 합니다.
                mask_image = init_image
                # 임시로 더미 마스크 지정 처리 예시

                # 4. 파이프라인 최종 실행 (Inference)
                result_image = self.pipe(prompt=prompt,
                                         image=init_image,
                                         mask_image=mask_image
                                         ).images[0]
                result_image_path = os.path.join(self.output_dir, img_name)
                result_image.save(result_image_path)
        # 다음 품목 학습을 위해 현재 품목 LoRA 가중치 언로드
        self.pipe.unload_lora_weights()