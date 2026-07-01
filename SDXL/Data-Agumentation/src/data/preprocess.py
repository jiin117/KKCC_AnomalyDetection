import os
import json
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration

def letter_padding_1024(raw_data_dir,
                        output_dir,
                        target_size=1024):
    for img_name in os.listdir(raw_data_dir):
        if not img_name.endswith('.png'): continue

        img_path = os.path.join(raw_data_dir, img_name)
        img = Image.open(img_path)

        # 비율 유지하며 리사이즈
        img.thumbnail((target_size, target_size))

        # 중앙 정렬 후 패딩 처리
        delta_w = target_size - img.width
        delta_h = target_size - img.height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        padded_img = ImageOps.expand(img, padding, fill='black') # 검정색 패딩
        padded_img.save(os.path.join(output_dir, img_name))

# if __name__ == '__main__':
#     letter_padding_1024("/home/ai-engr/KKCC/mvtec_ad_2/can/train/good",
#                         "/home/ai-engr/KKCC/Model_A_SDXL/0628/padding/can" )

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
            out = model.generate(**inputs, max_new_tokens = 50)
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

# if __name__ == "__main__":
#     sample = auto_Caption()
#     sample.generate_captions("/home/ai-engr/KKCC/Model_A_SDXL/0628/can_padded", "can")
#     sample.create_diffusers_metadata()