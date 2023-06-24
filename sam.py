import pathlib
import json
from typing import Union, Optional
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import cv2

from libs.sam_with_clip import load_image, SAMWithCLIP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', required=True, help='Input image folder')
    parser.add_argument('-o', '--output_dir', required=True, help='Path to save output mask')
    parser.add_argument('--mode', default='auto', help='SAM prediction mode: ["auto", "points"]')
    return parser.parse_args()


def segment_with_sam(sam_with_clip: SAMWithCLIP, img_path: Union[str, pathlib.Path]):
    img = load_image(img_path, mode='RGB')
    if sam_with_clip.sam_predictor_type == 'auto':
        masks = sam_with_clip.generate_masks(img)
    else:
        masks = sam_with_clip.generate_masks(img)
    _, _, merge_masks_list = sam_with_clip.find_overlap_masks(masks)

    crop = sam_with_clip.crop_image(img)
    similarity, class_indices = sam_with_clip.mask_classification(merge_masks_list, crop)
    return merge_masks_list, class_indices


def generate_point_grids(num_points_per_side: Union[int, tuple],
                         size: tuple,
                         negative_mask: Optional[np.ndarray] = None):
    w, h = size
    if isinstance(num_points_per_side, int):
        nx = ny = num_points_per_side
    else:
        nx, ny = num_points_per_side
    xs = np.linspace(0.01, 0.99, nx) * w
    ys = np.linspace(0.01, 0.99, ny) * h
    xv, yv = np.meshgrid(xs, ys)
    grid_points = np.stack((xv, yv), axis=2).reshape(-1, 2)
    if negative_mask is not None:
        assert h, w == negative_mask.shape
        points = []
        for x, y in grid_points:
            if negative_mask[int(y), int(x)] == 0:
                points.append((x, y))
        return np.array(points)
    return grid_points


def main():
    args = parse_args()
    if args.mode == 'points':
        negative_mask = cv2.imread("data/negative_masks.png")[:, :, 0]
        h, w = negative_mask.shape
        point_coords = generate_point_grids(32, (w, h), negative_mask)
        point_coords /= (w, h)  # SAM require point_grids normalize in range (0,1)
    else:
        point_coords = None
    sam_with_clip = SAMWithCLIP(sam_predictor_type=args.mode,
                                prompts=['cucumber', 'leaf', 'dark blob', 'number'],
                                point_coords=point_coords)
    image_dir = pathlib.Path(args.image_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    image_files = list(image_dir.glob('*jpg'))
    print('found {} images in {} folder'.format(len(image_files), image_dir))
    for imf in tqdm(image_files, total=len(image_files)):
        masks, class_indices = segment_with_sam(sam_with_clip, imf)
        out_result_dir = out_dir / imf.with_suffix("").name
        out_result_dir.mkdir()
        meta = []
        for i, m in enumerate(masks):
            dest = str(out_result_dir / '{}.png'.format(i))
            seg = m['segmentation']
            cv2.imwrite(dest, seg.astype(int) * 255)
            meta.append({'mask_file': dest, 'predicted_class': sam_with_clip.prompts[class_indices[i]]})

        with open(out_result_dir / 'label.json', 'w') as f:
            json.dump(meta, f)
        shutil.copy(imf, out_result_dir / imf.name)


if __name__ == '__main__':
    main()
