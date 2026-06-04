import os
import glob
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


class ImageAgu:
    def __init__(self, lora_path, edge_base_path="./screw_edges", output_base_path="./image_agu"):
        self.edge_base_path = edge_base_path
        self.output_base_path = output_base_path

        # 1. 모델 설정
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        # LoRA 로드
        self.pipe.load_lora_weights(lora_path, weight_name="adapter_model.safetensors")

    def _get_prompt_for_category(self, category):
        """if문으로 각 카테고리 이름과 일치하는 프롬프트 나오도록 수정 예정"""
        prompt = f"a high quality industrial photo of {category}, with deep scratches, severe dents, large dark oil stains, defective product, surface damage"
        negative_prompt = "clean, flawless, perfect condition, brand new, smooth surface, high quality"

        return prompt, negative_prompt

    def augment_category(self, category):
        """특정 카테고리에 대한 데이터 증강 실행"""
        cat_edge_folder = os.path.join(self.edge_base_path, category)
        cat_output_folder = os.path.join(self.output_base_path, category)
        os.makedirs(cat_output_folder, exist_ok=True)

        # 카테고리별 맞춤 프롬프트 호출
        prompt, negative_prompt = self._get_prompt_for_category(category)

        edge_paths = glob.glob(os.path.join(cat_edge_folder, "*.png"))
        print(f"{category} 증강 시작")

        for path in edge_paths:
            file_name = os.path.basename(path)
            save_path = os.path.join(cat_output_folder, f"aug_{file_name}")
            edge_image = Image.open(path).convert("RGB")

            # 난수 생성기로 다양성 확보
            generator = torch.Generator(device="cuda").manual_seed(torch.randint(0, 100000, (1,)).item())

            # 이미지 생성
            generated_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=edge_image,
                num_inference_steps=20,
                controlnet_conditioning_scale=0.8,
                guidance_scale=10.0,
                generator=generator
            ).images[0]

            generated_image.save(save_path)
            print(f"완료: {save_path}")

# [사용 예시]
# augmentor = ImageAgu(lora_path="/home/ai-engr/KKCC/ControlNet/my_factory_style/adapter_model.safetensors")
# augmentor.augment_category("screw")