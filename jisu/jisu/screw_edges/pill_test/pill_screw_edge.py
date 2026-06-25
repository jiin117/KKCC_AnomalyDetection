import cv2
import glob
import os
# 가우시안블러와 오츠이진화방식으로 외곽선을 딴 코드

# 1. input = MVTec AD pill 사진이 들어있는 폴더 경로 (good 데이터로 기준 잡기)
input_folder = "../../../../mvtec_ad_1/pill/train/good/"
# input_folder = "/home/ai-engr/KKCC/mvtec_ad_1/pill/train/good/"

# 2. output = 알약 외곽선 지도 저장할 폴더 만들기
output_folder = "LineArt/pillLineArt/"
os.makedirs(output_folder, exist_ok=True)

# 3. 폴더 안의 모든 .png 사진 파일 주소 가져오기
image_paths = glob.glob(os.path.join(input_folder, "*.png"))

print(f"총 {len(image_paths)}장의 알약 사진을 찾았습니다. 오츠 이진화 기반 선 따기를 시작합니다!")

# 4. 🔥 [반복문 시작]
for path in image_paths:
    file_name = os.path.basename(path)

    # 흑백(GrayScale)으로 이미지 로드
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # [전처리 단계 1] 가우시안 블러
    # 알약 표면의 미세한 흠집이나 내부 글자(노이즈)를 살짝 뭉개서
    # 오츠 알고리즘이 '배경 vs 알약 덩어리' 구조에만 집중하게 만듭니다.
    # blurred = cv2.GaussianBlur(src-대상 이미지, ksize-커널 또는 필터의 크기, sigmaX-x축 방향의 표준편차(흐림의 강도)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # [전처리 단계 2] 오츠 이진화 (Otsu's Binarization)
    # 컴퓨터가 알아서 최적의 임계값(otsu_thru)을 계산하고, 배경은 0(검은색), 알약은 255(흰색)로 만듭니다.
    # MVTec pill은 배경이 어두우므로 cv2.THRESH_BINARY를 씁니다.
    otsu_thru, thresh_image = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # 🌟 여기에 프롬프트(콘솔) 출력 함수 추가!
    print(f"[{file_name}] 적용된 오츠 임계값: {otsu_thru}")

    # [전처리 단계 3] 캐니 엣지로 외곽선 추출
    # 오츠 이진화를 거치면 이미지가 0과 255로만 이루어지기 때문에,
    # 캐니 엣지의 임계값(threshold1, threshold2)은 대형 노이즈를 거르는 수준(예: 100, 200)으로 주면
    # 오직 알약의 깨끗한 '테두리 선'만 아주 예쁘게 따집니다.
    edge_image = cv2.Canny(thresh_image, threshold1=100, threshold2=200)

    # 새로 만든 폴더에 저장
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, edge_image)

print("✨ 알약 외곽선 따기 작업이 완벽하게 끝났습니다!")