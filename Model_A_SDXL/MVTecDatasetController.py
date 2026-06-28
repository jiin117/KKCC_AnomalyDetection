import os
import gc
import torch

class MVTecDatasetController:
    """
    MVTec AD 데이터셋의 카테고리별 경로를 격리하여
    각 전처리 및 학습 모듈에 순차적으로 주입하는 메인 컨트롤러 클래스
    """
    def __init__(self, root_dir, padder, captioner, trainer, generator, script_path):
        self.root_dir = root_dir
        self.categories = None
        self.category = None
        self.raw_data_dir = os.path.join(root_dir, "mvtec_ad_2")
        self.train_target_dir = None
        self.lora_weights_dir = os.path.join(root_dir, "lora_weights")
        self.lora_weights_category_dir = None
        self.result_image_dir = os.path.join(root_dir, "result_images")
        self.padder = padder
        self.captioner = captioner
        self.trainer = trainer
        self.generator = generator
        self.script_path = script_path

    def _get_valid_categories(self):
        """루트 디렉토리에서 유효한 품목 폴더명만 추출"""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # 하위 항목 중 디렉토리만 필터링하여 리스트업
        return [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

    def run_one_category(self, category):
        """각 품목별로 독립적인 프로세스를 실행하는 제어 루프"""
        self.raw_data_dir = os.path.join(self.root_dir, category, "train/good")

        if not os.path.exists(self.raw_data_dir):
            print(f"[경고] {category}의 학습 경로({self.raw_data_dir})가 존재하지 않아 건너뜁니다.")

        try:
            print(f"[{category}] 1단계: 레터박스 패딩 전처리 시작")
            self.train_target_dir = os.path.join('./padded_img', category)
            os.makedirs(self.train_target_dir, exist_ok=True)
            self.padder(self.raw_data_dir, self.train_target_dir)

            print(f"[{category}] 2단계: 자동 캡셔닝 및 트리거 단어 삽입 시작")
            self.captioner.generate_captions(self.train_target_dir, category)
            self.captioner.create_diffusers_metadata()

            print(f"[{category}] 3단계: Diffusers 기반 SDXL LoRA 학습 및 가중치 도출 시작")
            self.trainer.extract_lora_weights(self.script_path, category, self.train_target_dir, self.lora_weights_dir)
            self.lora_weights_category_dir = self.trainer.get_lora_weights_category_dir()

            print(f"[{category}] 4단계: SAM 마스크와 LoRA 가중치를 활용하여 증강 이미지 생성 시작")
            self.generator.generate_image(self.lora_weights_category_dir,
                                          self.train_target_dir,
                                          self.result_image_dir,
                                          prompt = prompt) # prompt 입력 받는 함수 작성!

        except Exception as e:
            print(f"[{category}] 파이프라인 수행 중 에러 발생: \n{e}")

        finally:
            # 메모리 누수 방지를 위한 매 카테고리 종료 후 리소스 정리 (VRAM 내 세션 해제)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f'[{category}] 리소스 정리 완료')

    def run_all_categories(self):
        self.categories = self._get_valid_categories()
        for category in self.categories:
            self.category = category
            self.raw_data_dir = os.path.join(self.root_dir, self.category, "train/good")

            if not os.path.exists(self.raw_data_dir):
                print(f"[경고] {self.category}의 학습 경로({self.raw_data_dir})가 존재하지 않아 건너뜁니다.")
                continue

            self.run_one_category(self.category)