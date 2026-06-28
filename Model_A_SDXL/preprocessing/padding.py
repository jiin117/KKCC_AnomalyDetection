from PIL import Image, ImageOps
import os

def letter_padding_1024(raw_data_dir,
                        output_dir,
                        target_size=1024):
    for img_name in os.listdir(raw_data_dir):
        if not img_name.endswith('.png'): continue

        img_path = os.path.join(raw_data_dir, img_name)
        img = Image.open(img_path)

        # 비율 유지하며 리사이즈
        img.thumbnail((target_size, target_size))

        # 중앙 정렬 후 패딩 처리
        delta_w = target_size - img.width
        delta_h = target_size - img.height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        padded_img = ImageOps.expand(img, padding, fill='black') # 검정색 패딩
        padded_img.save(os.path.join(output_dir, img_name))

# if __name__ == '__main__':
#     letter_padding_1024("/home/ai-engr/KKCC/mvtec_ad_2/can/train/good",
#                         "/home/ai-engr/KKCC/Model_A_SDXL/0628/padding/can" )