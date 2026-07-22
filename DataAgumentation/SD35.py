
    self.PaddingImages.image_paths
    if torch.cuda.is_available():
        device = "cuda"
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        generated_image_path = set_generated_image_path(self.object)
        for idx in range(RawDataset.__len__(self.RawDataset)):
            padding_image = self.PaddingImages.get_image(idx)
            generated_image = pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=padding_image,
                height=1024,
                width=1024,
                num_inference_steps=50,
            ).images[0]
            generated_image_name = "gen" + self.object + "_" + str(idx).zfill(3) + ".png"
            image_path = os.path.join(generated_image_path, generated_image_name)
            generated_image.save(image_path)
            print(f"{self.object} : 인덱스 {idx + 1}번 증강이미지 생성 완료")