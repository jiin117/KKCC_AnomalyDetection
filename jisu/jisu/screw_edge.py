import cv2
import glob
import os

# 1. input = 원본 MVTec AD 2 사진 들어있는 폴더 경로 지정
input_folder = "D:/KKCC/mvtec_ad_2/fabric/train/good"
# 2. output = 외곽선 지도 저장할 폴더 만들기
output_folder = "./screw_edges/fabric_test"
os.makedirs(output_folder, exist_ok=True)  # 폴더 없으면 자동 생성

# 3. [핵심] 폴더 안의 모든 .png 사진 파일들의 주소를 전부 리스트로 반환
image_paths = glob.glob(os.path.join(input_folder, "00[0-9]_regular.png"))

print(f"총 {len(image_paths)}장의 사진을 찾았습니다. 자동 선 따기를 시작합니다!")

# 4. 🔥 [반복문 시작] 사진 개수만큼 컴퓨터가 알아서 뺑뺑이를 돕니다.
for path in image_paths:
    # 파일 이름만 쏙 추출합니다 (예: '000.png')
    file_name = os.path.basename(path)

    # 이미지를 흑백으로 읽어옵니다.
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 캐니 엣지로 외곽선을 따버립니다.
    edge_image = cv2.Canny(image, threshold1=100, threshold2=170)

    # 새로 만든 폴더에 같은 이름으로 저장합니다 (예: './screw_edges/000.png')
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, edge_image)

print("✨ 수백 장의 사진 외곽선 따기 작업이 완벽하게 끝났습니다!")