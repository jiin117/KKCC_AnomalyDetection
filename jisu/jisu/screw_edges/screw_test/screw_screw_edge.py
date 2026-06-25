import cv2
import glob
import os

# 1. input 폴더 경로 (mvtec_ad_1 / screw)
input_folder = "D:/KKCC/mvtec_ad_1/screw/train/good"

# 2. output 폴더 만들기
output_folder = "D:/kkcc/LineArt/ScrewLineart"
os.makedirs(output_folder, exist_ok=True)

# 3. 폴더 안의 모든 .png 파일 가져오기
image_paths = glob.glob(os.path.join(input_folder, "*.png"))
print(f"총 {len(image_paths)}장의 screw 사진을 찾았습니다. 작업을 시작합니다!")

# 4. 반복문 시작
for path in image_paths:
    file_name = os.path.basename(path)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue

    # [1단계] 가우시안 블러
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # [2단계] 오츠 이진화
    otsu_thru, thresh_image = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # [3단계] 캐니 엣지
    edge_image = cv2.Canny(thresh_image, threshold1=100, threshold2=200)

    # 저장
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, edge_image)

    # 콘솔창에 컴퓨터가 찾은 오츠 임계값 출력
    print(f"[{file_name}] 오츠 임계값: {otsu_thru}")

print("✨ Screw 외곽선 따기 작업 완료!")