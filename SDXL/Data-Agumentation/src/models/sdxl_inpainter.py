# preprocessing/inference.py
import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

"""증강 이미지를 생성하는 모듈"""
class SDXLInpaintGenerator:
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
            variant="fp16",
            use_safetensors=True
        ).to(self.device)
        print("=> [SDXLInpaintGenerator] 모델 로드 완료.")

        self.category = None
        self.lora_dir = None
        self.input_dir = None
        self.output_dir = None
        self.prompt = None

    def generate_image(self,
                       lora_dir,
                       padded_dir,
                       mask_dir,
                       output_dir,
                       prompt):
        """특정 품목의 LoRA를 로드하여 마스크 기반 인페인팅 이미지를 생성하는 메서드"""
        self.lora_dir = lora_dir
        self.padded_dir = padded_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.prompt = prompt
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(self.lora_dir):
            self.pipe.load_lora_weights(self.lora_dir)
            print(f" => [SDXLInpaintGenerator] {os.path.basename(self.lora_dir)} 결합 완료.")
        else:
            print("Lora 가중치 결합 실패")

        for img_name in os.listdir(self.padded_dir):
            if img_name.lower().endswith('.png'):
                padded_img_path = os.path.join(self.padded_dir, img_name)
                mask_img_path = os.path.join(self.mask_dir, img_name)
                init_image = Image.open(padded_img_path).convert("RGB")
                mask_image = Image.open(mask_img_path).convert("RGB")

                # 4. 가이드 문서 규격 준수 이미지 생성 (cross_attention_kwargs 반영)
                with torch.no_grad():
                    result_image = self.pipe(
                        prompt=prompt,
                        image=init_image,
                        mask_image=mask_image,
                        num_inference_steps=10,
                        num_images_per_prompt=1,
                        cross_attention_kwargs={"scale": 0.8}  # LoRA 가중치 강도 조절
                    ).images[0]

                # 결과 저장
                result_image_path = os.path.join(self.output_dir, f"gen_{img_name}")
                result_image.save(result_image_path)
                torch.cuda.empty_cache()
        # 다음 품목 학습을 위해 현재 품목 LoRA 가중치 언로드
        self.pipe.unload_lora_weights()

if __name__ == "__main__":
    sample = SDXLInpaintGenerator()
    prompt1 = ("A pristine can, smooth and clean texture, even lighting, "
               "highly detailed, realistic industrial photography, 8k resolution")
    sample.generate_image("/home/ai-engr/KKCC/Model_A_SDXL/0628/pytorch_lora_weights.safetensors",
                       "/home/ai-engr/KKCC/Model_A_SDXL/0628/can_padded",
                       "/home/ai-engr/KKCC/Model_A_SDXL/0628/can_mask",
                       "/home/ai-engr/KKCC/Model_A_SDXL/0628/can_gen/normal",
                       prompt1)
    prompt2 = ("A can with a noticeable dent, physical shape deformation, "
               "damaged surface, industrial defect inspection")
    sample.generate_image("/home/ai-engr/KKCC/Model_A_SDXL/0628/pytorch_lora_weights.safetensors",
                       "/home/ai-engr/KKCC/Model_A_SDXL/0628/can_padded",
                       "/home/ai-engr/KKCC/Model_A_SDXL/0628/can_mask",
                       "/home/ai-engr/KKCC/Model_A_SDXL/0628/can_gen/abnormal",
                       prompt2)