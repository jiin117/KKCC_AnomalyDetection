import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import gc

print("🚀 [2번 실행 파일] 정품 LoRA 뇌를 장착하고 이미지 초고속 양산 시작...")

# 1. 외곽선 감시용 자(ControlNet_github_original) 로드
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

# 2. 메인 AI 엔진 빌려와서 그래픽카드(CUDA)에 탑재
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# ⭕ [100% 정답 연동 구역]
# 1번 파일이 완벽하게 구워낸 정품 폴더의 절대경로 주소를 따서 그대로 꽂아 넣습니다.
lora_dir = os.path.abspath("D:/KKCC/ControlNet_images/ControlNet_diffusersLibrary/"
                                "my_factory_style/adapter_model.safetensors")
# ★★★★ 아까 가중치 파일 생성한 거 경로 넣어주심 됩니다!!!
pipe.load_lora_weights(lora_dir, weight_name = "adapter_model.safetensors")

# 3. 형태를 잡아줄 기준선 이미지 (나사/캔 외곽선)
# (경로가 다를 경우 질문자님의 진짜 000_regular.png 사진 주소로만 살짝 바꿔주세요!)
url = ("D:/KKCC/ControlNet_images/ControlNet_diffusersLibrar"
       "y/screw_edges/fabric_test/000_regular.png") # ★★★★★외곽선 따준 파일 있죠! 그거 하나 집어서 경로 넣어주세요

hint_image = load_image(url)

# 4. 생성 프롬프트 주문서
prompt = "industrial defect, deep scratches, torn edges, crushed areas, severe dents, damaged packaging, surface deformation"
# 프롬프트 추가해야함.

# 5. 진짜 훈련 데이터를 토대로 3초 만에 이미지 대량 생성!
generated_image = pipe(prompt, image=hint_image).images[0]

# 6. 결과 저장
save_folder = "./results"
os.makedirs(save_folder, exist_ok=True)
final_path = os.path.join(save_folder, os.path.basename(url))
generated_image.save(final_path) # ★★★★★ 생성한 이미지 파일명 지정해주세요!
print("🎉 프로젝트 폴더에 생성된 사진이 저장되었습니다!")

# 마감 청소
torch.cuda.empty_cache()
gc.collect()

exit()
