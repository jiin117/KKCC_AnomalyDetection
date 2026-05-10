"""
This script verifies your submission for completeness and checks for common
formatting errors. Run this script before uploading your submission to the
evaluation server to avoid reductions in your evaluation budget due to minor
mistakes. When all checks were successful, the submission directory is
compressed.
이 스크립트는 귀하가 제출한 결과물의 완전성을 검증하고, 흔히 발생하는 형식 오류를 점검합니다.
사소한 실수로 인해 평가 횟수(Budget)가 차감되는 것을 방지하기 위해,
평가 서버에 제출물을 업로드하기 전 이 스크립트를 반드시 실행하십시오. 모든 검사항목이 성공적으로 통과되면,
제출용 디렉터리는 자동으로 압축됩니다.
"""

# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
'''
명령행(Command Line)에서 인자를 입력받기 위한 라이브러리입니다. 
사용자가 터미널에서 특정 폴더 경로 등을 지정할 수 있게 합니다.
'''
from pathlib import Path
'''
파일 및 디렉터리 경로를 객체로 다루는 라이브러리입니다. 
운영체제(OS)에 상관없이 경로 연산을 안전하게 수행합니다.
'''

from utils import (
    DIRECTORY_STRUCTURE,
    SubmissionException,
    check_anomaly_image_dir,
    check_images,
    compare_found_vs_required,
    compress_submission,
    logger,
)
'''
DIRECTORY_STRUCTURE : 제출물이 갖추어야 할 표준 디렉터리와 파일 구조 정보가 담긴 상수입니다.
SubmissionException : 검증 과정에서 오류(예: 파일 누락)가 발생했을 때 호출되는 사용자 정의 예외 클래스입니다.
check_anomaly_image_dir : 이상치 탐지(Anomaly Detection)를 위한 이미지 디렉터리가 규격에 맞게 구성되었는지 점검합니다.
check_images : 이미지 파일의 확장자, 손상 여부, 또는 해상도 등 이미지 데이터의 유효성을 검사합니다.
compare_found_vs_required : 실제 발견된 파일 목록과 필수 요구 파일 목록을 대조하여 누락 여부를 확인합니다.
compress_submission : 모든 검증을 통과한 제출물 디렉터리를 서버 업로드용 파일(.zip 등)로 압축합니다.
logger : 검증 과정 및 결과(성공, 경고, 오류)를 터미널이나 파일에 기록하는 객체입니다.
'''


def check_submission(submission_file_path: str) -> None:
    """
    Checks the structure and content of the submission directory.

    Args:
        submission_file_path (str): Path to the submission directory.

    Raises:
        SubmissionException: If directory structure or content is incorrect.
    """
    # 입력된 경로가 실제 디렉터리인지 확인합니다. 아닐 경우 SubmissionException을 발생시킵니다.
    logger.info(f"Start checking submission {submission_file_path}")
    submission_file_path = Path(submission_file_path)
    if not Path(submission_file_path).is_dir():
        raise SubmissionException("The given path is not a directory")

    # 필수 폴더인 anomaly_images가 있는지 확인하고, 내부의 .tiff 파일 형식과 이미지 상태를 검증합니다.
    required_ad_image_dirs = {'anomaly_images'}

    root_ad_images = submission_file_path / 'anomaly_images'
    if not root_ad_images.exists():
        raise SubmissionException(
            f"{root_ad_images.as_posix()} was not found. Please adhere to the "
            f"directory structure: \n{DIRECTORY_STRUCTURE}"
        )

    logger.info("Check structure and content of the anomaly_image directory")
    ad_image_paths = check_anomaly_image_dir(
        root_ad_images,
        expected_file_format='.tiff',
    )
    check_images(ad_image_paths, thresholded=False)
    logger.info("Done")

    # 선택 사항인 이진화 이미지 폴더를 검사합니다.
    # 폴더가 없으면 경고(Warning)와 함께 기준점 계산 방식(baseline)을 안내하고,
    # 폴더가 있으면 .png 형식을 검증합니다.
    root_thresh_ad_images = submission_file_path / 'anomaly_images_thresholded'
    if not root_thresh_ad_images.exists():
        logger.warning(
            f"{root_thresh_ad_images.as_posix()} was not found. To "
            f"evaluate threshold-dependent metrics, binarize the anomaly "
            f"images: Set normal pixels to 0 and anomalous pixels to 255. You "
            f"could start with this baseline method: segmentation_threshold = "
            f"np.mean(anomaly_scores_val) + 3 * np.std(anomaly_scores_val)"
        )
    else:
        logger.info(
            "Check structure and content of the anomaly_images_thresholded "
            "directory"
        )
        thresh_ad_image_paths = check_anomaly_image_dir(
            root_thresh_ad_images, expected_file_format='.png'
        )
        check_images(thresh_ad_image_paths, thresholded=True)
        logger.info("Done")

        required_ad_image_dirs.add('anomaly_images_thresholded')

    # 실제 발견된 폴더 세트와 필수 요구 세트를 대조하여 최종적으로 구조적 결함이 없는지 확인합니다.
    compare_found_vs_required(
        required_ad_image_dirs, set(), submission_file_path
    )
    logger.info("All checks successful")


if __name__ == "__main__":
    # 사용자가 터미널에서 입력한 제출 디렉터리 경로(submission_path)를 읽어옵니다.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "submission_path",
        type=str,
        help=(
            "Path to your submission directory containing all MVTec AD 2 "
            "objects."
        ),
    )
    args = parser.parse_args()
    # 앞서 정의한 check_submission 함수를 실행하여 검사를 시작합니다.
    check_submission(args.submission_path)
    # 검증이 성공적으로 완료(오류 없이 종료)되면, 해당 폴더를 서버 업로드용으로 압축합니다.
    compress_submission(args.submission_path)
