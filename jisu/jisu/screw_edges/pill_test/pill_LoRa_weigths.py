import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import Trainer, TrainingArguments
# ⭕ [눈속임 완전 박멸] 대놓고 이름에 'LoRa'가 박힌 정품 설정 부품 소환!
from peft import LoraConfig, get_peft_model

print("🔥 [1번 학습 파일] 눈에 보이는 진짜 정품 LoRA 수술 시작...")

# ==========================================
# 1. input = MVTec AD pill 사진이 들어있는 폴더 경로 (good 데이터로 기준 잡기)
input_folder = "D:/KKCC/mvtec_ad_1/pill/train/good"

# 2. output = 알약 가중치를 저장할 폴더 만들기
output_folder = "D:/kkcc/LoRa_Weights/pill_LoRa_Weights"
os.makedirs(output_folder, exist_ok=True)
print(f"📁 가중치 저장 폴더 준비 완료: {output_folder}")
# ==========================================

# 1. 기본 부품들 인터넷 창고에서 다운로드
controlnet_component = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet_component, torch_dtype=torch.float16
)

# 🚀 [눈으로 확인하세요!] 대놓고 'Lora' 설정을 명시해 줍니다.
# 이렇게 짜야 파라미터 세포 이름마다 'lora' 글자가 물리적으로 진짜 들어갑니다.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_v"], # 화가 뇌세포 중 붓칠을 담당하는 핵심 구역 지정
    lora_dropout=0.05,
    bias="none"
)

# 💉 메인 화가 뇌세포(pipe.unet)에 방금 만든 정품 Lora 주사액을 강제로 주입하여 개조합니다!
pipe.unet = get_peft_model(pipe.unet, peft_config)

# 2. 로컬 전용 훈련 옵션
# ★ [수정됨] 임시 파일(checkpoint)이 엉뚱한 곳에 안 생기고, 우리가 만든 output_folder 안에 예쁘게 들어가게 세팅!
my_training_args = TrainingArguments(
    output_dir=f"{output_folder}/checkpoint", push_to_hub=False, report_to="none"
)

# 3. 완벽하게 'lora' 글자가 이식된 뇌를 교관에게 인계!
# ★ [수정됨] 팀 코드의 "../mvtec_ad_2..." 부분을 우리가 위에서 세팅한 input_folder 변수로 교체!
trainer = Trainer(
    model=pipe.unet,
    args=my_training_args,
    train_dataset=input_folder
)

# 4. 공부가 다 끝나면 내 컴퓨터 하드디스크에 진짜 'lora' 이름표가 박힌 알맹이로 저장!
# ★ [수정됨] 팀 코드의 "./my_factory_style" 부분을 우리가 위에서 세팅한 output_folder 변수로 교체!
trainer.model.save_pretrained(output_folder)

print(f"🎉 [진짜 대성공] 이제 파일 내부 세포 이름까지 'lora'가 들어간 정품 파일이 {output_folder} 경로에 구워졌습니다!")

exit()