import json
import pathlib
import numpy as np
import cv2
import imutils
from tqdm import tqdm
from libs.medial_axis_analysis import calc_skeleton_line, chord_length_parameterization_method


def find_bbox_center(binary_mask):
    cnts = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 1:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnt = cnts[0]
    x, y, w, h = cv2.boundingRect(cnt)
    # cx, cy = x + w / 2, y + h / 2
    # return cx, cy
    return x, y


def measure_medial_axis(mask_file: str):
    binary_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    # Compute the skeleton
    skeleton, coords = calc_skeleton_line(binary_mask)
    medial_axis_length = chord_length_parameterization_method(coords)
    return binary_mask, skeleton, medial_axis_length

def main():
    root_dir = pathlib.Path("libs/SuperGluePretrainedNetwork/agri_phenotyping/masks_SAM_and_CLIP")
    mask_folders = list(root_dir.glob("*"))
    scale_factor = 9.54 # pixels/cm

    for f in tqdm(mask_folders, total=len(mask_folders)):
    # for mask_loc in mask_folders:
        mask_files = list(mask_loc.glob("*.png"))
        xs = []
        ys = []
        measurements = []
        try:
            for mf in mask_files:
                binary_mask, skeleton, medial_axis_length = measure_medial_axis(str(mf))
                cx, cy = find_bbox_center(binary_mask)
                skel_file_name = "skel_" + mf.with_suffix('.jpg').name
                cv2.imwrite(str(mask_loc / skel_file_name), skeleton.astype(np.uint8) * 255)
                xs.append(cx)
                ys.append(cy)
                measurements.append(medial_axis_length / scale_factor)  # to cm

            indexes = list(range(len(mask_files)))
            indexes.sort(key=xs.__getitem__)
            records = [{'mask_file': mask_files[i].name,
                        'length': measurements[i],
                        'x': xs[i], 'y': ys[i]} for i in indexes]

            with open(mask_loc / "measurements.json", 'w') as f:
                json.dump(records, f, indent=2)
        except BaseException as e:
            print("Error with:", mf)


if __name__ == '__main__':
    main()
