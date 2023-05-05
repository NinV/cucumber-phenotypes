import pathlib
import argparse
from tqdm import tqdm
import numpy as np
# import cv2.cv2 as cv
import cv2 as cv


def parse_args():
    parser = argparse.ArgumentParser(description='Homography transformation')
    parser.add_argument(
        '--input_pairs', type=str, default='agri_phenotyping/pairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='agri_phenotyping',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '-m', '--match_dir', default='agri_phenotyping/out',
        help='Path to the directory that contains match results (.npz)')
    parser.add_argument(
        '-o', '--out_dir', default='agri_phenotyping/warped',
        help='Path to the directory that contains match results (.npz)')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    return parser.parse_args()


def read_image(path, resize, resize_float=False):
    image = cv.imread(str(path))
    if image is None:
        return None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv.resize(image, (w_new, h_new)).astype('float32')

    return image, (w_new, h_new)


def validate_size(opt):
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def get_matches(match_dir: pathlib.Path,
                src_img_file: pathlib.Path,
                target_image_file: pathlib.Path,):

    stem0, stem1 = src_img_file.stem, target_image_file.stem
    matches_path = match_dir / '{}_{}_matches.npz'.format(stem0, stem1)
    matches_data = np.load(str(matches_path))   # ['keypoints0', 'keypoints1', 'match_confidence', 'matches']

    src_kps = matches_data['keypoints0']
    target_kps = matches_data['keypoints1']
    src_to_target_matches = matches_data['matches']

    matched_indices = np.where(src_to_target_matches != -1)[0]

    matched_src_kps = src_kps[matched_indices]
    matched_target_kps = target_kps[src_to_target_matches[matched_indices]]

    return matched_src_kps, matched_target_kps


def main():
    args = parse_args()
    validate_size(args)
    root_dir = pathlib.Path(args.input_dir)
    match_dir = pathlib.Path(args.match_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_pairs, 'r') as f:
        pairs = [l.strip().split(',') for l in f.readlines()]

    # print(pairs)

    for src_img_file, target_image_file in tqdm(pairs):
        src_img_file = root_dir / src_img_file
        target_image_file = root_dir / target_image_file
        src_img, src_size = read_image(src_img_file, args.resize, args.resize_float)   # read image and resize
        target_img, target_size = read_image(target_image_file, args.resize, args.resize_float)

        src_kps, target_kps = get_matches(match_dir, src_img_file, target_image_file)

        H, _ = cv.findHomography(src_kps, target_kps)
        warped = cv.warpPerspective(src_img, H, target_size)
        saved_warped = out_dir / '{}_{}'.format('warped', src_img_file.name)
        cv.imwrite(str(saved_warped), warped)


if __name__ == '__main__':
    main()
