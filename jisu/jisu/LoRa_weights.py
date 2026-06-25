# screw_edge로 외곽선 따고 가중치 뽑아내는 코드
# screw_edge-> LoRa_weights.py -> imageAgu.py(실행파일)
# 이 파일을 돌려서 나오는 임계값을 기록할것!(나중에 발표나 보고서에 활용할 예정)
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import Trainer, TrainingArguments
# ⭕ [눈속임 완전 박멸] 대놓고 이름에 'Lora'가 박힌 정품 설정 부품 소환!
from peft import LoraConfig, get_peft_model

print("🔥 [1번 학습 파일] 눈에 보이는 진짜 정품 LoRA 수술 시작...")

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
my_training_args = TrainingArguments(
    output_dir="./my_factory_style_checkpoint", push_to_hub=False, report_to="none"
) # ★★★★output_dir 경로는 편한대로 만드세요!

# 3. 완벽하게 'lora' 글자가 이식된 뇌를 교관에게 인계!
trainer = Trainer(
    model=pipe.unet,
    args=my_training_args,
    train_dataset="../mvtec_ad_2/can/train" #★★★★★ 당신이 학습시키고자하는 데이터가 들어있는 폴더 경로로 수정하세요!!!!!
)

# 4. 공부가 다 끝나면 내 컴퓨터 하드디스크에 진짜 'lora' 이름표가 박힌 알맹이로 저장!
trainer.model.save_pretrained("./my_factory_style") #★★★★★★ 당신의 학습시켜서 생성한 가중치 파일이 들어갈 폴더 경로로 수정하세요1!!
print("🎉 [진짜 대성공] 이제 파일 내부 세포 이름까지 'lora'가 들어간 정품 파일이 구워졌습니다!")

exit()