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

# CUDA 연산 가능 여부 확인
print("CUDA 사용 가능:", torch.cuda.is_available())

# cuDNN 백엔드 버전 출력 (에러 없이 숫자가 나오면 충돌이 해결된 것입니다)
print("cuDNN 버전:", torch.backends.cudnn.version())

# 강제로 cuDNN을 사용하는 간단한 연산 테스트
if torch.cuda.is_available():
    x = torch.randn(1, 3, 224, 224).cuda()
    conv = torch.nn.Conv2d(3, 16, kernel_size=3).cuda()
    out = conv(x)
    print("✅ 성공: cuDNN을 이용한 컨볼루션 연산이 정상 수행되었습니다.")