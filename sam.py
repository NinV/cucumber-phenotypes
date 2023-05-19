import pathlib
import json
from typing import Union
import argparse
from tqdm import tqdm
import cv2

from libs.sam_with_clip import load_image, SAMWithCLIP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', required=True, help='Input image folder')
    parser.add_argument('-o', '--output_dir', required=True, help='Path to save output mask')
    return parser.parse_args()


def segment_with_sam(sam_with_clip: SAMWithCLIP, img_path: Union[str, pathlib.Path]):
    img = load_image(img_path, mode='RGB')
    masks = sam_with_clip.generate_masks(img)
    _, _, merge_masks_list = sam_with_clip.find_overlap_masks(masks)

    crop = sam_with_clip.crop_image(img)
    similarity, class_indices = sam_with_clip.mask_classification(merge_masks_list, crop)
    return merge_masks_list, class_indices


def main():
    args = parse_args()
    sam_with_clip = SAMWithCLIP(crop=(55, 310, 506, 720))
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


if __name__ == '__main__':
    main()
