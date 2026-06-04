import datetime
import cv2
import os
import glob

class Preprocess:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        current_date = datetime.now().strftime("%y%m%d")
        self.output_path = os.path.join(output_path, current_date)
        self.categories = [d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d))]

    def fetch_image(self, category):
        input_folder = os.path.join(self.dataset_path, category, 'train', 'good')
        if not os.path.isdir(input_folder):
            return
        image_paths = glob.glob(os.path.join(input_folder, "*.png"))
        for path in image_paths:
            file_name = os.path.basename(path)
            img = cv2.imread(path)
            if img is not None:
                yield file_name, img

    def fabric(self, img):
        """
        직물(fabric): 짜임새와 부드러운 무늬가 날아가지 않도록 Canny 임계값을 대폭 낮춤
        """
        return cv2.Canny(img, 20, 60)

    def metal_nut(self, img):
        """
        금속 너트(metal_nut): 빛 반사가 심하고 형태가 뚜렷하므로 노이즈를 억제하는 기본/높은 임계값
        """
        return cv2.Canny(img, 100, 200)

    def bottle(self, img):
        """
        병(bottle): 유리 재질의 투명도와 굴절을 잡기 위한 중간 임계값 적용
        """
        return cv2.Canny(img, 50, 150)

    def _default_processor(self, img):
        """
        전용 함수가 정의되지 않은 새로운 카테고리가 들어왔을 때 사용할 기본 방어 코드
        """
        return cv2.Canny(img, 100, 200)

    """실행"""
    def run(self):
        for category in self.categories:
            output_folder = os.path.join(self.output_path, category)
            os.makedirs(output_folder, exist_ok=True)

            # 파이썬 마법: category 문자열(예: 'fabric')과 똑같은 이름의 함수를 self 안에서 찾음!
            # 만약 직접 정의한 함수가 없다면 _default_processor를 대신 꽂아줌.
            process_func = getattr(self, category, self._default_processor)

            print(f"[{category}] 품목 전처리 시작 (적용 함수: {process_func.__name__})")
            for file_name, img in self.fetch_image(category):
                # 찾은 맞춤형 전처리 함수 실행!
                edges = process_func(img)

                # 결과물 저장
                save_path = os.path.join(output_folder, file_name)
                cv2.imwrite(save_path, edges)

        print("모든 카테고리의 ControlNet 가이드라인 추출 완료")


# 파일이 직접 실행될 때만 아래 로직 작동 (다른 곳에서 import 할 때 튕김 방지)
if __name__ == "__main__":
    DATASET_PATH = "/home/ai-engr/KKCC/test"
    OUTPUT_PATH = "/home/ai-engr/KKCC/DataAgu/guidelineIMG"

    # 클래스 소환 및 실행
    preprocessor = Preprocess(DATASET_PATH, OUTPUT_PATH)
    preprocessor.run()