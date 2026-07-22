from process import Processing
dataset_name = ['MVTEC_AD_1', 'MVTEC_AD_2']

MVTEC_AD_1_OBJECTS = [ 'bottle', 'cable', 'capsule', 'carpet', 'grid',
                       'hazelnut', 'leather', 'metal nut', 'pill',
                       'screw', 'tile', 'toothbrush', 'transistor',
                       'wood', 'zipper', ]

MVTEC_AD_2_OBJECTS = ['can', 'fabric', 'fruit_jelly', 'rice',
                      'sheet_metal', 'vial', 'wallplugs', 'walnuts', ]
PROMPT = ["프롬프트 추가하기, 최소 10개"]

if __name__ == "__main__":
    for object_1 in MVTEC_AD_1_OBJECTS:
        object_name = object_1
        preprocessor = Processing(dataset_name='MVTEC_AD_1', object_name=object_name, )
        # preprocessor.padding_all_images_in_object()
        preprocessor.generate_image_SDXL()
    for object_2 in MVTEC_AD_2_OBJECTS:
        object_name = object_2
        preprocessor = Processing(dataset_name='MVTEC_AD_2', object_name=object_name, )
        # preprocessor.padding_all_images_in_object()
        preprocessor.generate_image_SDXL()



