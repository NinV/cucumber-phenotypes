#!/bin/bash
ROOT_DIR="agri_phenotyping"
TEMPLATE_FILE="board_template_images/A1-v1/A1-0004-v1.jpg"
IMG_DIR="fruits"
IMG_PAIRS="pairs.txt"
RESIZE="800"
MATCH_THRES="0.9"
MATCH_OUT_DIR="${ROOT_DIR}/match"
WARP_OUT_DIR="${ROOT_DIR}/warped"

python prepare_pairs.py -r "${ROOT_DIR}" -t "${TEMPLATE_FILE}" -i "${IMG_DIR}" -o "${ROOT_DIR}/${IMG_PAIRS}"
echo -e "######################### Template / Img pairs #########################"
head "${ROOT_DIR}/${IMG_PAIRS}"

python match_pairs.py --input_pairs "${ROOT_DIR}/${IMG_PAIRS}" --input_dir "${ROOT_DIR}" \
  --output_dir "${MATCH_OUT_DIR}" --resize "${RESIZE}"  --match_threshold "${MATCH_THRES}"

python warp.py --input_pairs "${ROOT_DIR}/${IMG_PAIRS}" --input_dir "${ROOT_DIR}" \
  -m "${MATCH_OUT_DIR}" --resize "${RESIZE}" -o "${WARP_OUT_DIR}"
