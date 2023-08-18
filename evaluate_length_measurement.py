import argparse
import json
import pathlib
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
from libs.medial_axis_analysis import CucumberShape
from libs.vis_utils import draw_point, draw_line, draw_polygon, draw_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', required=True, help='Path to input image folder')
    parser.add_argument('--prefix', default='warped_', help='')
    parser.add_argument('-l', '--label_file', required=True, help='Path to label file (csv)')
    parser.add_argument('-o', '--output', default='out.json', help='save result to json file')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--peak_thresh', type=float, default=0.3)
    parser.add_argument('--max_stem_width', type=float, default=0.2)
    return parser.parse_args()


def analyze_error(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    length_prediction_with_gt = []
    errors = []
    for original_image_prediction in data.values():
        mask_folder = pathlib.Path(original_image_prediction['mask_folder'])
        for obj in original_image_prediction['objects']:
            gt, pred = obj['gt'], obj['pred']
            length_prediction_with_gt.append([str(mask_folder / obj['file_name']), gt, pred, (pred - gt) / gt])
            errors.append((abs(pred - gt) / gt))

    length_prediction_with_gt.sort(key=lambda item: abs(item[3]), reverse=True)
    error_5 = np.abs(np.array(errors) < 0.05).sum() / len(errors) * 100
    print('Mean absolute error (MAE) / AE < 5%: {:.4f}/{:.2f}'.format(np.mean(errors), error_5))
    return length_prediction_with_gt


def match_labels_with_predictions(image_dir, labels, args):
    label_to_prediction = {}
    for file_name, lengths in labels.items():
        mask_folder = image_dir / (args.prefix + file_name)
        with open(mask_folder / 'label.json', 'r') as f:
            seg_prediction = json.load(f)
        seg_prediction.sort(key=lambda item: (item['bbox'][0] + item['bbox'][2]) / 2)   # sort bbox from left to right

        mask_files = []
        for mask_data in seg_prediction:
            if mask_data['predicted_class'] == 'cucumber':
                mask_file_name = pathlib.Path(mask_data['mask_file']).name
                mask_files.append(mask_folder / mask_file_name)
        if len(mask_files) != len(lengths):
            print('[Warning] Number of objects in labels file does not match prediction ({} != {}) at:{}'.format(
                len(mask_files), len(lengths), mask_folder))
            print('################################################\n')
            continue
        label_to_prediction[file_name] = mask_files
    return label_to_prediction


def calc_length(mask_file, args, num_samples=200, cm_per_pixel=0.1048, visualize=False):
    binary_mask_file = pathlib.Path(mask_file)
    binary_mask = cv2.imread(str(binary_mask_file), cv2.IMREAD_GRAYSCALE)
    analyzer = CucumberShape()
    tck, _, _, boundary = analyzer.find_medial_axis(binary_mask)

    # calculate width and remove stem
    profiles = analyzer.width_profile(tck, boundary, num_samples=num_samples,
                                      peak_thresh=args.peak_thresh, max_stem_width=args.max_stem_width)

    if visualize:
        chart_and_gif_files_prefix = 'width_profile_'
        stem_point_prefix = 'stem_point_'
        curve_points = profiles['curve_points']
        analyzer.plot_chart(profiles, conversion_scale=cm_per_pixel, unit='cm',
                            save_loc=binary_mask_file.parent / (
                                        chart_and_gif_files_prefix + binary_mask_file.with_suffix('.pdf').name))

        # Define the desired height and width
        h, w = binary_mask.shape
        # Create an empty image
        image = Image.new("RGB", (w, h))
        draw_polygon(image, boundary, fill_color=0)
        draw_line(image, curve_points)

        image_gif = []
        for normal_line, cp, length in zip(profiles['normal_lines'], profiles['curve_points'],
                                           profiles['cumulative_lengths']):
            clone = image.copy()
            draw_line(clone, normal_line, color=(0, 255, 255))
            draw_point(clone, cp)
            draw_text(clone, (10, 10), 'width: {:.2f} cm'.format(normal_line.length * cm_per_pixel), font_size=36)
            draw_text(clone, (10, 50), 'length: {:.2f} cm'.format(length * cm_per_pixel), font_size=36)
            image_gif.append(clone)
        image_gif[0].save(
            binary_mask_file.parent / (chart_and_gif_files_prefix + binary_mask_file.with_suffix('.gif').name),
            save_all=True, append_images=image_gif[1:], optimize=False,
            duration=10, loop=0)
        peak_img = image_gif[profiles['peak_idx']]
        peak_img.save(binary_mask_file.parent / (stem_point_prefix + binary_mask_file.with_suffix('.png').name))

    return profiles['curve_length (remove stem)'] * cm_per_pixel


def main():
    args = parse_args()
    df = pd.read_csv(args.label_file, header=0, usecols=['file_name', 'length'])
    labels = df.groupby('file_name')['length'].apply(list).to_dict()
    image_dir = pathlib.Path(args.image_dir)
    label_to_prediction = match_labels_with_predictions(image_dir, labels, args)

    meta = {}
    for i, mask_files in tqdm(label_to_prediction.items()):
        objects_ = []
        for gt, mf in zip(labels[i], mask_files):
            objects_.append({'file_name': mf.name, 'gt': gt, 'pred': calc_length(mf, args, visualize=args.viz)})
        meta[i] = {'mask_folder': str(mask_files[0].parent), 'objects': objects_}

    with open(args.output, 'w') as f:
        json.dump(meta, f, indent=2)
    analyze_error(args.output)


if __name__ == '__main__':
    main()
