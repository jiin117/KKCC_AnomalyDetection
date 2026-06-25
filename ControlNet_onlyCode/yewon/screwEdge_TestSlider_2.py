import cv2
import numpy as np


# 트랙바 조작 시 호출될 빈 함수
def nothing(x):
    pass


# 테스트할 이미지 1장의 경로를 입력해 (네 환경에 맞게 수정)
image_path = r"D:\KKCC_Project\mvtec_ad_1\capsule\train\good\000.png"

# 이미지를 흑백으로 불러오기
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지를 불러오지 못했습니다. 경로를 확인해 주세요!")
    exit()

# 'Canny Simulator'라는 이름의 창 생성
cv2.namedWindow('Canny Simulator', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Canny Simulator', 800, 800)  # 창 크기 넉넉하게 조절

# 슬라이더(트랙바) 생성
# cv2.createTrackbar('트랙바 이름', '창 이름', 기본값, 최댓값, 콜백함수)
cv2.createTrackbar('Blur Size', 'Canny Simulator', 2, 10, nothing)  # 0~10 조절 (홀수로 변환됨)
cv2.createTrackbar('Lower (Min)', 'Canny Simulator', 40, 255, nothing)  # 하한선
cv2.createTrackbar('Upper (Max)', 'Canny Simulator', 90, 255, nothing)  # 상한선

print("시뮬레이터가 실행되었습니다. ESC 키를 누르면 종료됩니다.")

while True:
    # 1. 트랙바의 현재 값들을 실시간으로 읽어오기
    blur_val = cv2.getTrackbarPos('Blur Size', 'Canny Simulator')
    lower = cv2.getTrackbarPos('Lower (Min)', 'Canny Simulator')
    upper = cv2.getTrackbarPos('Upper (Max)', 'Canny Simulator')

    # 블러 필터 크기는 반드시 '홀수'여야 함 (1, 3, 5, 7, 9 ...)
    ksize = blur_val * 2 + 1

    # 2. 가우시안 블러 적용 (배경 노이즈 뭉개기)
    if ksize > 1:
        blurred_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:
        blurred_img = img.copy()  # 블러 값이 0(ksize=1)이면 원본 사용

    # 3. 현재 이미지의 v값(Median) 계산
    v = int(np.median(blurred_img))-120

    # 4. 읽어온 하한/상한선 값으로 Canny 외곽선 추출
    edged = cv2.Canny(blurred_img, lower, upper)

    # 5. 화면에 텍스트(현재 값들)를 표시하기 위해 흑백을 컬러(BGR) 캔버스로 변환
    display_img = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    # 상단에 v값, 하한선, 상한선, 블러 크기 출력
    info_text = f"v(Median): {v} | Lower: {lower} | Upper: {upper} | Blur: {ksize}x{ksize}"
    cv2.putText(display_img, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 6. 결과 화면 출력
    cv2.imshow('Canny Simulator', display_img)

    # ESC 키(27)를 누르면 무한 루프 탈출 후 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 모든 창 닫기
print(f"v: {v}\n,lower: {lower}\n,upper: {upper}\n,BlurKsize: {ksize}, ")

cv2.destroyAllWindows()