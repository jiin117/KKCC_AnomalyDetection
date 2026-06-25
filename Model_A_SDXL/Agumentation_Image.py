import torch
from diffusers import StableDiffusionXLInpaintPipeline
#
# # 1. SDXL Inpainting 베이스 파이프라인 로드
# pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
#     torch_dtype=torch.float16,
#     variant="fp16"
# ).to("cuda")

# # 2. 직접 생성한 LoRA 가중치 파일 경로 연결
# lora_weights_dir = "/home/ai-engr/KKCC/Model_A_SDXL/lora_weights"
# pipe.load_lora_weights(lora_weights_dir)

# 3. SAM 이미지 분할 로직 실행 및 마스크(Mask) 이미지 획득 (코드 생략)
# init_image = 원본 이미지
# mask_image = SAM이 생성한 흑백 마스크 이미지

# 4. 이미지 생성 실행 (Inference)
# cross_attention_kwargs를 통해 LoRA의 강도(scale)를 조절할 수 있습니다.
prompt = "A photo of [학습한 LoRA 트리거 단어], highly detailed, 4k"
result_image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=30,
    cross_attention_kwargs={"scale": 0.8} # LoRA 가중치 적용 비율 (0.0 ~ 1.0)
).images[0]

result_image.save("output_result.png")

