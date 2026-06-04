import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

class LoRA_weights:
    def __init__(self, category, base_data_path="../mvtec_ad_2"):
        self.category = category
        self.base_data_path = base_data_path
        self.output_dir = f"./LoRA/weights/{self.category}"

        # 모델 및 파이프라인 초기화
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        )

        # PEFT 설정
        self.peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["to_q", "to_v"],
            lora_dropout=0.05,
            bias="none"
        )
        self.pipe.unet = get_peft_model(self.pipe.unet, self.peft_config)

    def prepare_dataset(self):
        # MVTec AD 구조에 따른 데이터 경로 설정
        dataset_path = f"{self.base_data_path}/{self.category}/train"
        print(f"데이터셋 로드 경로: {dataset_path}")
        return dataset_path

    def train(self):
        print(f"{self.category} 학습 시작!")

        train_args = TrainingArguments(
            output_dir=f"./LoRA/training_args/{self.category}",
            push_to_hub=False,
            report_to="none"
        )

        # 데이터 로직 연결
        trainer = Trainer(
            model=self.pipe.unet,
            args=train_args,
            train_dataset=self.prepare_dataset()
        )

        trainer.train()

        # 카테고리별 디렉터리에 저장
        trainer.model.save_pretrained(self.output_dir)
        print(f"{self.category} 가중치 저장 완료: {self.output_dir}")

# [사용 예시]
# trainer = LoRA_weights(category="bottle")
# trainer.train()




