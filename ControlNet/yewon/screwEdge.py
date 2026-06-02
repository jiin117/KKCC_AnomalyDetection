import cv2
import os
import glob
import numpy as np

"""MVTec AD 데이터셋 전체 불러오기"""
dataset_path = r"D:\KKCC_Project\mvtec_ad_1\cable\train\good"
output_path = r"D:\KKCC_Project\mvtec_1_edge\cable_edge3"
# os.makedirs(output_path, exist_ok=True)

# dataset_path 폴더의 하위 폴더 목록 리스트로 반환
categories = os.listdir(dataset_path)
print("카테고리 수:", len(categories))

#❗❗❗❗❗❗❗이 부분만 복사 후 고쳐 사용!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############################################################################### 
# 1. 빛을 고르게 + 알아서 외곽선 따기 (Python 코드)
# 이 코드는 두 단계를 거칩니다. 첫째, 어둡거나 너무 밝은 사진의 조명을 평범하게 맞추고(CLAHE),
# 둘째, 사진의 평균 밝기를 계산해 스스로 임계값을 정해서 외곽선을 땁니다(Auto Canny).
def auto_canny_with_clahe(image_path):
    # 1. 이미지를 흑백으로 불러오기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. CLAHE 적용 (조명과 대비를 균일하게 알아서 맞춰줌)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced_img = clahe.apply(img)

    # 3. 이미지의 '중간 밝기값(Median)' 찾기
    v = np.median(balanced_img)

    # 4. 중간 밝기값을 기준으로 하한선/상한선 임계값 자동 계산 (기본 33% 여유를 둠)
    sigma = 0.44
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # 5. 계산된 임계값으로 Canny 외곽선 추출
    edged = cv2.Canny(balanced_img, lower, upper)

    return edged
#❗❗❗❗❗❗❗여기까지#################################################################################

# 카테고리 별로 가져옴
for category in categories:
    # input_folder = os.path.join(dataset_path, category, 'train', 'good')
    input_folder = r"D:\KKCC_Project\mvtec_ad_1\cable\train\good"
    if not os.path.isdir(input_folder):
        continue
    # 사진 주소 경로 리스트로 반환 -> 현재 디렉터리의 모든 .png 파일들
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))[:10]
    # 저장도 카테고리 별로
    output_folder = r"D:\KKCC_Project\mvtec_1_edge\cable_edge3\3T"
    os.makedirs(output_folder, exist_ok=True)

    # 사진 한 장씩 edge 따서 저장
    for path in image_paths:
        file_name = os.path.basename(path)
        # 1) 이미지 읽기
        img = cv2.imread(path)


        # ##################################################
        # #외곽선 작업 전 옵션 추가#
        #
        #
        # # 흑백 변환 - 외곽선 검출 연산의 효율 증대
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        # # 적응형 임계값 처리 - 조명이 고르지 않은 이미지에 가장 효과적
        # # 주변 영역의 밝기 기준으로 임계값 계산해 빛 영향 줄여 윤곽선 추출
        # thresh = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY, 7, 2)
        #
        # # 모폴로지 연산 (Morphology)
        # # 빛 때문에 외곽선이 끊기거나 점처럼 분산된 부분 메우고(팽창) 노이즈 제거(침식)
        # kernel = np.ones((5, 5), np.uint8)
        # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # #########################################################

        # result_image = auto_canny_with_clahe("test_image.png")

        # 2) Canny 외곽선 추출 (임계값 품목별로 다르게 해야함)
        # edges = cv2.Canny(img, 100, 200)
        edges = result_image = auto_canny_with_clahe(path)


        # 3) 결과 저장
        save_path = os.path.join(output_folder, file_name)
        cv2.imwrite(save_path, edges)
# print(output_folder)

print("edge 추출 완료!")
