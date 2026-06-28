import torch
import torchvision

print("Torch 버전:", torch.__version__)
print("Torchvision 버전:", torchvision.__version__)

# nms 연산자 호출 테스트 (에러 없이 출력되면 정상 해결된 것입니다)
try:
    print("NMS 연산자 상태:", torchvision.ops.nms)
    print("✅ 성공: torchvision::nms 연산자가 정상적으로 로드되었습니다.")
except RuntimeError as e:
    print("❌ 실패:", e)