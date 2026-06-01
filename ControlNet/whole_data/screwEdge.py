import cv2
import os
import glob

"""MVTec AD 데이터셋 전체 불러오기"""
dataset_path = "/home/ai-engr/KKCC/test"
output_path = "./screw_edges"

# dataset_path 폴더의 하위 폴더 목록 리스트로 반환
categories = os.listdir(dataset_path)
print("카테고리 수:", len(categories))

# 카테고리 별로 가져옴
for category in categories:
    input_folder = os.path.join(dataset_path, category, 'train', 'good')
    if not os.path.isdir(input_folder):
        continue
    # 사진 주소 경로 리스트로 반환 -> 현재 디렉터리의 모든 .png 파일들
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    # 저장도 카테고리 별로
    output_folder = os.path.join(output_path, category)
    os.makedirs(output_folder, exist_ok=True)

    # 사진 한 장씩 edge 따서 저장
    for path in image_paths:
        file_name = os.path.basename(path)
        # 1) 이미지 읽기
        img = cv2.imread(path)
        # 2) Canny 외곽선 추출 (임계값 품목별로 다르게 해야함)
        edges = cv2.Canny(img, 100, 200)
        # 3) 결과 저장
        save_path = os.path.join(output_folder, file_name)
        cv2.imwrite(save_path, edges)

print("edge 추출 완료!")