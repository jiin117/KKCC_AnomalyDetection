import os
import glob
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ControlNet 가져
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

# Stable Diffusion 가져와 GPU(CUDA) 탑재
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# LoRA 가중치 만든 거 사용
lora_path = "/home/ai-engr/KKCC/ControlNet/my_factory_style/adapter_model.safetensors"
pipe.load_lora_weights(lora_path, weight_name = "adapter_model.safetensors")

# edge 폴더 & 증강 이미지를 저장 폴더
edge_path = "./screw_edges"
output_path = "./image_agu"

# 폴더 내의 카테고리 리스트 가져오기
categories = os.listdir(edge_path)
print(f"카테고리 개수: {len(categories)}")

# 🔥 [2단계] 폴더(품목)별로 순회하며 이미지 대량 생성 시작
for category in categories:
    cat_edge_folder = os.path.join(edge_path, category)
    # 폴더 아니면 건너뛰기
    if not os.path.isdir(cat_edge_folder):
        continue

    # 결과물을 저장할 품목별 폴더 자동 생성
    cat_output_folder = os.path.join(output_path, category)
    os.makedirs(cat_output_folder, exist_ok=True)

    # 해당 품목 폴더 안의 모든 외곽선 사진(.png) 주소 수집
    edge_paths = glob.glob(os.path.join(cat_edge_folder, "*.png"))

    # 프롬프트 자동화: 품목 이름을 프롬프트에 동적으로 삽입 -> 프롬프트 내용은 바꿔야 함!
    prompt = f"a high quality industrial photo of {category}, with deep scratches, severe dents, large dark oil stains, defective product, surface damage"
    negative_prompt = "clean, flawless, perfect condition, brand new, smooth surface, high quality"

    # 사진 한 장씩 이미지 생
    for path in edge_paths:
        file_name = os.path.basename(path)

        # 증강된 이미지 이름 설정 (구분을 위해 앞에 'aug_' 붙임)
        save_name = f"aug_{file_name}"
        save_path = os.path.join(cat_output_folder, save_name)

        # 외곽선 이미지 불러오기 (ControlNet은 PIL Image 형식을 선호합니다)
        edge_image = Image.open(path).convert("RGB")

        # [다양성 확보] 매 사진마다 난수(Seed)를 다르게 주어 결함 모양을 무작위로 만듦
        random_seed = torch.randint(0, 100000, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(random_seed)

        # AI 이미지 생성!
        generated_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=edge_image,  # 방금 따온 외곽선 도면 입력
            num_inference_steps=20,  # 생성 디테일 (20~30 적당)
            controlnet_conditioning_scale=0.8,  # ★ 외곽선 통제력 (0.8로 낮춰야 결함이 생길 숨통이 트임)
            guidance_scale=10.0,  # ★ 프롬프트(결함) 집착도 (높을수록 불량을 강하게 그림)
            generator=generator
        ).images[0]

        # 저장
        generated_image.save(save_path)
        print(f"완료: {save_name} (Seed: {random_seed})")

print("모든 품목의 데이터 증강 완료!")