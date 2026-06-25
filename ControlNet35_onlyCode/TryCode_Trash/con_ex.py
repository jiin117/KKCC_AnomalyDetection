###### 260608(월) 현재 ControlNet_onlyCode 안에 있는 코드들 라이브러리 StableDiffusion3Pipeline 로 바꾼 후 실행해보겠습니다.


import torch
import os
from diffusers import StableDiffusion3Pipeline


#모델 사용하기 위한 access 토큰 할당
os.environ["HF_TOKEN"] = "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"
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo")


# 모델 ID 설정 (Large 모델 기준)
# SD 3.5 Large 모델: "stabilityai/stable-diffusion-3.5-large"
# SD 3.5 Large Turbo 모델: "stabilityai/stable-diffusion-3.5-large-turbo"
model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

# 파이프라인 로드 (VRAM 16GB 이하인 경우 torch_dtype=torch.float16 설정 권장)
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda") # GPU 메모리에 할당

# 테스트할 프롬프트 및 네거티브 프롬프트
# prompt = "A cinematic shot of a cute red panda wearing a tiny wizard hat, reading a spellbook in a mystical forest, highly detailed, 8k"
# negative_prompt = "blurry, cropped, ugly, distorted, deformed"
prompt = "Hyper-detailed, photorealistic macro photograph of a dense, tightly packed, uniformly layered field of raw, \
        uncooked short-grain white rice kernels, filling the entire frame. \
        Each individual kernel is distinctly rendered with varied translucency and opacity, showing a natural mix of off-white, \
        cream, and pure white tones. The surface has a subtle, natural sheen. \
        Soft, diffused overhead lighting evenly illuminates the texture without harsh shadows, highlighting the individual rounded, slightly oblong shapes. \
        The focus is critically sharp across the central area, with a subtle depth of field fall-off towards the far edges. \
        Tiny natural imperfections, chalky spots, and variations are visible. Direct overhead view, flat lay composition. \
        8k resolution, ultra-high definition, natural food photography style."

negative_prompt = "cooked rice, steamed rice, sticky rice, wet, moisture, \
                    long-grain rice, basmati rice, jasmine rice, brown rice, black rice, \
                    yellow grains, foreign objects, bugs, insects, dirt, stones, hands, spoon, bowl, \
                    plate, background, container edge, harsh shadows, high contrast, low resolution, \
                    blurry, deformed kernels, fused grains, plastic texture, \
                    3D render, CGI, cartoon, illustration, drawing, painting, unnatural color gradients."

# 이미지 생성 (가이드 단계와 해상도 설정)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=28, # Large 모델 기준 권장 수치 (Turbo는 4~8)
    guidance_scale=4.5,     # 프롬프트 가이드 강도
).images[0]

# 생성된 이미지 저장
image.save("./sd35_test_image.png")
print("이미지가 'sd35_test_image_쌀.png'로 저장되었습니다.")

