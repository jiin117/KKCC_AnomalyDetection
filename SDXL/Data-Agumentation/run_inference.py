

if __name__ == "__main__":
    # 디렉토리 경로
    root_dir = "/"
    script_path = "train_text_to_image_lora_sdxl.py"

    # 클래스 인스턴스화
    captioner = auto_Caption()
    trainer = LoraWeightExtractor()
    generator = SDXLInpaintGenerator()

    # 파이프라인 실행
    manager = MVTecDatasetController(
        root_dir,
        padder=letter_padding_1024,
        captioner=captioner,
        trainer=trainer,
        generator=generator,
        script_path = script_path
    )

    # MVTec ad 2 categories = 'wallplugs', 'vial', 'rice', 'can', 'walnuts', 'fabric', 'fruit_jelly', 'sheet_metal'
    manager.run_one_category('vial')  # 선택한 카테고리 1개만 실행
    # manager.run_all_categories()  # 모든 카테고리 실행