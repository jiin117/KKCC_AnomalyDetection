import os
import glob
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

import sys
import diffusers
import transformers
from diffusers.utils import is_peft_available
# import peft
print(sys.version)
print(sys.executable)
print(diffusers.__version__)
print(transformers.__version__)
print(is_peft_available())
# print(peft.__version__)


print("🪄 [마지막 3단계] 외곽선(뼈대) + 가중치(물감) + 프롬프트(명령) 퓨전 시작!")

# ==========================================
# 📁 1. [초보자 맞춤형 경로 설정]
# 아까 우리가 1, 2단계에서 정성껏 만들었던 파일들의 위치를 정확히 꽂아줍니다!

# 재료 1: 외곽선 도면들이 있는 방 (1단계 결과물)
edge_path = "D:/kkcc/LineArt/pillLineArt"

# 재료 2: 물감(가중치)이 있는 방과 그 안의 파일 이름 (2단계 결과물)
lora_folder = "D:/kkcc/LoRA_Weights/pill_LoRa_Weights"
lora_file_name = "adapter_model.safetensors"

# 최종 결과물: 가짜 불량 알약 사진들이 대량으로 저장될 완전히 새로운 방!
output_path = "D:/kkcc/Augmented_Images/pill_Defects"
os.makedirs(output_path, exist_ok=True)
print(f"📁 최종 불량품 저장 폴더 준비 완료: {output_path}")
# ==========================================

# 2. 외곽선을 꽉 잡아줄 ControlNet_onlyCode 부품 가져오기
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

# 3. 그림을 그릴 화가 AI (Stable Diffusion) 데려오기 + 강력한 GPU 장착!
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 4. 화가 AI에게 우리가 아까 만든 '알약 전용 가중치(물감)' 쥐어주기
pipe.load_lora_weights(lora_folder, weight_name=lora_file_name)


# 5. 외곽선 방에 있는 알약 도면들 싹 다 불러오기
edge_paths = glob.glob(os.path.join(edge_path, "*.png"))
print(f"총 {len(edge_paths)}장의 알약 도면을 찾았습니다. 본격적인 합성을 시작합니다!")


# 6. 🗣️ 마법의 주문서 (프롬프트 설정)
# prompt(긍정 주문): "알약에 깨짐, 금, 오염 같은 심각한 불량을 그려줘!"
prompt = "a high quality industrial photo of a pill, broken pill, cracked surface, chipped edges, color stains, contamination, defective product"

# negative_prompt(부정 주문): "깨끗하고 완벽한 새 상품처럼은 절대 그리지 마!"
negative_prompt = "clean, flawless, perfect condition, brand new, smooth surface, high quality"


# 7. 🔥 도면을 하나씩 꺼내서 진짜 가짜 불량품으로 만들기!
for path in edge_paths:
    file_name = os.path.basename(path) # 원래 이름 (예: 000.png)
    save_name = f"aug_{file_name}"     # 헷갈리지 않게 앞에 'aug_'를 붙입니다.
    save_path = os.path.join(output_path, save_name)

    # 도면 이미지 열기
    edge_image = Image.open(path).convert("RGB")

    # [다양성 확보] 매번 다른 모양의 결함이 생기도록 무작위 주사위(Seed) 굴리기
    random_seed = torch.randint(0, 100000, (1,)).item()
    generator = torch.Generator(device="cuda").manual_seed(random_seed)

    # 💥 화가 AI야, 그림을 그려라!
    generated_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edge_image,             # 뼈대 (1단계 외곽선)
        num_inference_steps=20,       # 붓칠 횟수 (20번이 기본)
        controlnet_conditioning_scale=0.8, # ★ 외곽선 지키는 강도 (0.8로 여유를 줌)
        guidance_scale=10.0,          # ★ 내 주문서(프롬프트) 말을 강박적으로 듣는 강도
        generator=generator
    ).images[0]

    # 완성된 그림 저장
    generated_image.save(save_path)
    print(f"✅ 완성: {save_name} (Seed: {random_seed})")

print("🎉 대성공!! 모든 가짜 불량 알약 생성이 끝났습니다! 저장 폴더를 확인해 보세요!")