import os
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class auto_Caption:
    def __init__(self):
        pass
    def generate_captions(self, train_target_dir, trigger_word):
        self.train_target_dir = train_target_dir
        # 1. 모델 로드 (GPU 가속 활용)
        device = "cuda"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

        for img_name in os.listdir(self.train_target_dir):
            if not img_name.endswith(('.png')): continue

            img_path = os.path.join(self.train_target_dir, img_name)
            raw_image = Image.open(img_path).convert('RGB')

            # 캡션 생성
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # 파일 저장 (Trigger Word + AI 생성 캡션)
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            with open(os.path.join(self.train_target_dir, txt_name), "w") as f:
                f.write(f"{trigger_word}, {caption}")  # 트리거 단어 강제 삽입

    def create_diffusers_metadata(self):
        # 기존 .txt 파일들을 통합하여 metadata.jsonl 생성 (diffusers 규격)
        metadata_path = os.path.join(self.train_target_dir, "metadata.jsonl")
        with open(metadata_path, "w", encoding="utf-8") as f_jsonl:
            for file in os.listdir(self.train_target_dir):
                if file.endswith(".txt") and file != "metadata.jsonl":
                    base_name = os.path.splitext(file)[0]
                    img_name = base_name + ".png"
                    if os.path.exists(os.path.join(self.train_target_dir, img_name)):
                        # 캡션 내용 읽기
                        with open(os.path.join(self.train_target_dir, file), "r", encoding="utf-8") as f_txt:
                            caption = f_txt.read().strip()

                        # jsonl 한 줄 작성
                        line = {"file_name": img_name, "text": caption}
                        f_jsonl.write(json.dumps(line, ensure_ascii=False) + "\n")
                        break