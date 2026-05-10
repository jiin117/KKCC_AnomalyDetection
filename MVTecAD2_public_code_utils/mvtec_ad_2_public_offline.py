"""
MVTec AD 2 Dataset.
The dataset class provides functions to load images (and ground truth if
applicable) from the MVTec AD 2 dataset and to store anomaly images in the
correct structure for evaluating performance on the evaluation server.

MVTec AD 2 데이터셋 관련 클래스입니다.
본 데이터셋 클래스는 MVTec AD 2 데이터셋으로부터 이미지(및 해당하는 경우
정답 데이터인 Ground Truth)를 로드하는 기능을 제공합니다. 또한, 평가 서버에서
성능을 측정할 수 있도록 이상치 이미지(Anomaly Images)를 올바른 디렉터리 구조로
저장하는 기능을 포함하고 있습니다.
"""

# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0

import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor

PATH_TO_MVTEC_AD_2_FOLDER = 'PATH_TO_.../mvtec_ad_2'
MVTEC_AD_2_OBJECTS = [
    'can',
    'fabric',
    'fruit_jelly',
    'rice',
    'sheet_metal',
    'vial',
    'wallplugs',
    'walnuts',
]


class MVTecAD2(Dataset):
    """
    Dataset class for MVTec AD 2 objects.
    Args:
        mad2_object (str): can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts
        split (str): train, validation, test_public, test_private, test_private_mixed
        transform (function, optional): transform applied to samples, defaults to 'to_tensor'
    Args:
        mad2_object (str): 대상 객체 종류 (can, fabric, fruit_jelly, rice,
                            sheet_metal, vial, wallplugs, walnuts 중 선택)
        split (str): 데이터 분할 모드 (train, validation, test_public,
                        test_private, test_private_mixed)
        transform (function, optional): 샘플에 적용할 변환 함수 (기본값: 'to_tensor')
    """

    def __init__(
        self,
        mad2_object,
        split,
        transform=to_tensor,
    ):
        """
        중괄호 { }를 사용하여 집합(Set) 자료형을 만들었습니다.
        split 변수의 값이 집합 내의 5가지 문자열 중 하나라도 포함되어 있는지 확인합니다.
        리스트([ ])보다 집합({ })을 사용하면 탐색 속도가 더 빠르기 때문에 효율적인 작성 방식입니다.

        만약 split에 정의되지 않은 값(예: 'test_v2')이 들어오면 AssertionError를 발생시킵니다.
        이때 어떤 잘못된 값이 들어왔는지 f-string을 통해 명확하게 출력하여 디버깅을 돕습니다.
        """
        assert split in {
            'train',
            'validation',
            'test_public',
            'test_private',
            'test_private_mixed',
        }, f'unknown split: {split}'

        assert (
            mad2_object in MVTEC_AD_2_OBJECTS
        ), f'unknown MVTec AD 2 object: {mad2_object}'

        self.object = mad2_object
        self.split = split
        self.transform = transform

        self._image_base_dir = PATH_TO_MVTEC_AD_2_FOLDER

        self._object_dir = os.path.join(self._image_base_dir, mad2_object)
        # MVTecAD2 데이터셋 클래스의 초기화 과정에서 특정 객체(Object)의 이미지가 저장된 물리적 경로를 설정
        # get all images from the split
        self._image_paths = sorted(glob.glob(self._get_pattern()))
        """
        데이터셋 내의 이미지 파일 경로들을 찾아내어 정렬된 리스트 형태로 저장
        sorted(...): 파일명을 알파벳/숫자 순서로 정렬하여 데이터의 일관성을 보장
        glob.glob(...): 인자로 받은 패턴과 일치하는 모든 파일의 경로를 리스트로 반환
        self._get_pattern(): 이미지 파일들을 찾기 위한 검색 패턴(와일드카드 문자열)을 
                            예) "/data/can/train/good/*.png"와 같은 형태의 문자열을 생성
                            
        <이 작업이 필요한 이유>
        컴퓨터(AI 모델)는 폴더에 사진이 몇 장 있는지 자동으로 알지 못함.
        - 전체 수량 파악: len 함수 사용 가능
        - 순번 부여: 리스트에 담기면 self._image_paths[0]은 첫 번째 사진, 
            self._image_paths[1]은 두 번째 사진이라는 식으로 인덱스(번호)를 통해 접근가능
        - 데이터 로딩 스케줄링: 모델이 "5번 사진 가져와!"라고 요청하면, 
            이 리스트의 5번 경로를 보고 실제 하드디스크에서 이미지를 읽어옴
        """

    def _get_pattern(self) -> str:
        if 'private' in self.split:
        # 'test_private', 'test_private_mixed'의 사진들을 의미
            return os.path.join(
                self._object_dir, self.split, '[0-9][0-9][0-9]*.png'
            )

        # '[0-9][0-9][0-9]*.png'의 의미
        # [0-9]: 0부터 9 사이의 숫자 하나
        # [0-9][0-9][0-9]: 숫자가 연속으로 3개 나오는 것
        # *: 그 뒤에는 어떤 글자가 와도 상관없다
        # .png: 확장자는 반드시 .png

        return os.path.join(
            self._object_dir,
            self.split,
            '[gb][oa][od]*',  # bad나 good 폴더의 사진을 불러오라는 의미
            '[0-9][0-9][0-9]*.png',
        )

    # 사진 개수 세는 것
    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get dataset item for the index ``idx``.

        Args:
            idx (int): Index to get the item.

        Returns:
            dict[str,  str | torch.Tensor]: Dict containing the sample image,
            image path, and the relative anomaly image output path for both
            image types continuous and thresholded.
        """

        image_path = self._image_paths[idx]
        sample = default_loader(image_path)
        if self.transform is not None:
            sample = self.transform(sample)

        return {
            'sample': sample,
            'image_path': image_path,
            'rel_out_path_cont': self.get_relative_anomaly_image_out_path(idx),
            'rel_out_path_thresh': self.get_relative_anomaly_image_out_path(
                idx, True
            ),
        }

    @property
    def image_paths(self):
        return self._image_paths

    @property
    def has_segmentation_gt(self) -> bool:
        return self.split == 'test_public'

    def get_relative_anomaly_image_out_path(self, idx, thresholded=False):
        """Returns a path relative to the experiment directory
        for storing the (thresholded) anomaly image in the required structure.

        Args:
            idx (int): sample index
            thresholded (bool): return output path for thresholded image,
            defaults to 'False'

        Returns:
            str: relative output path to write anomaly image
        """

        image_path = Path(self._image_paths[idx])
        relpath = image_path.relative_to(self._image_base_dir)

        if not thresholded:
            base_dir = 'anomaly_images'
            suffix = '.tiff'
        else:
            base_dir = 'anomaly_images_thresholded'
            suffix = '.png'

        return os.path.join(base_dir, relpath.with_suffix(suffix))

    def get_gt_image(self, idx):
        """Returns the ground truth image where values of 255 denote
        anomalous pixels and values of 0 anomaly-free ones. For good images
        'None' is returned.
        In case no segmentation ground truth is available
        (test_private/test_private_mixed) 'None' is returned as well.

        Args:
            idx (int): sample index

        Returns:
            numpy.array or None: ground truth image if available
        """
        gt_image = None
        if (
            self.has_segmentation_gt
            and 'good' not in self.get_relative_anomaly_image_out_path(idx)
        ):
            image_path = self.image_paths[idx]
            base_path, file_name = image_path.split('/bad/')
            gt_image_path = os.path.join(
                base_path, 'ground_truth/bad', file_name
            ).replace('.png', '_mask.png')

            gt_image_pil = Image.open(gt_image_path)
            gt_image = np.asarray(gt_image_pil)

        return gt_image
