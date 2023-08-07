import pathlib
import cv2
import numpy as np
import shapely
from PIL import Image
from libs.medial_axis_analysis import CucumberShape
from libs.vis_utils import draw_point, draw_line, draw_polygon, draw_text
from matplotlib import pyplot as plt


def plot_chart(curve_length, widths, save_loc='plot.png'):
    gradient_width = np.zeros(widths.shape[0])
    gradient_width[1:] = np.abs(widths[1:] - widths[0:-1])
    plt.plot(curve_length, widths, label='width (cm)')
    plt.plot(curve_length, gradient_width, label='width gradient (cm)')
    plt.xlabel('cumulative length (cm)')
    plt.ylabel('cm')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.legend()
    plt.savefig(save_loc)


def draw_skeleton(boundary: shapely.Polygon, skeleton: np.ndarray,
                  boundary_color=(255, 255, 255),  skeleton_color=(0, 0, 255),
                  thickness=1):
    # h, w = skeleton.shape
    # image = Image.new("RGB", (w, h))
    # draw_polygon(image, boundary, fill_color=None, outline_color=boundary_color, width=thickness)

    image = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    vertices = np.array(boundary.exterior.coords, dtype=np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    cv2.polylines(image, [vertices], isClosed=True, color=boundary_color, thickness=thickness)
    image[skeleton > 0] = skeleton_color
    return image


def draw_medial_axis(boundary: shapely.Polygon, img_shape, curve_points: np.ndarray,
                     boundary_color=(255, 255, 255),  axis_color=(0, 255, 0), point_color=(255, 0, 0),
                     thickness=1):
    h, w = img_shape
    image = Image.new("RGB", (w, h))
    draw_polygon(image, boundary, fill_color=None, outline_color=boundary_color, width=thickness)
    draw_line(image, curve_points, color=axis_color)
    draw_point(image, curve_points[0], color=point_color)
    draw_point(image, curve_points[-1], color=point_color)
    return image


def visualize(mask_file):
    binary_mask_file = pathlib.Path(mask_file)
    chart_and_gif_files_prefix = 'width_profile_'
    stem_point_prefix = 'stem_point_'

    binary_mask = cv2.imread(str(binary_mask_file), cv2.IMREAD_GRAYSCALE)
    analyzer = CucumberShape()
    tck, skeleton, _, boundary = analyzer.find_medial_axis(binary_mask)
    skeleton_viz = draw_skeleton(boundary, skeleton)
    cv2.imwrite(str(binary_mask_file.parent / ('skel' + binary_mask_file.with_suffix('.png').name)), skeleton_viz)
    cm_per_pixel = 0.1048
    num_samples = 200

    # calculate width and remove stem
    profiles = analyzer.width_profile(tck, boundary, num_samples=num_samples)
    curve_points = profiles['curve_points']
    medial_axis_viz = draw_medial_axis(boundary, binary_mask.shape, curve_points)
    medial_axis_viz.save(str(binary_mask_file.parent / ('medial_' + binary_mask_file.with_suffix('.png').name)))
    analyzer.plot_chart(profiles, conversion_scale=cm_per_pixel, unit='cm',
                        save_loc=binary_mask_file.parent / (chart_and_gif_files_prefix + binary_mask_file.with_suffix('.pdf').name))

    # Define the desired height and width
    h, w = binary_mask.shape
    # Create an empty image
    image = Image.new("RGB", (w, h))
    draw_polygon(image, boundary, fill_color=0)
    draw_line(image, curve_points)

    image_gif = []
    for normal_line, cp, length in zip(profiles['normal_lines'], profiles['curve_points'], profiles['cumulative_lengths']):
        clone = image.copy()
        draw_line(clone, normal_line, color=(0, 255, 255))
        draw_point(clone, cp)
        draw_text(clone, (10, 10), 'width: {:.2f} cm'.format(normal_line.length * cm_per_pixel), font_size=36)
        draw_text(clone, (10, 50), 'length: {:.2f} cm'.format(length * cm_per_pixel), font_size=36)
        image_gif.append(clone)
    image_gif[0].save(binary_mask_file.parent / (chart_and_gif_files_prefix + binary_mask_file.with_suffix('.gif').name),
                      save_all=True, append_images=image_gif[1:], optimize=False,
                      duration=10, loop=0)
    peak_img = image_gif[profiles['peak_idx']]
    peak_img.save(binary_mask_file.parent / (stem_point_prefix + binary_mask_file.with_suffix('.png').name))

    print('Length with stem (cm):', profiles['curve_length'] * cm_per_pixel)
    print('Length w/o stem (cm):', profiles['curve_length (remove stem)'] * cm_per_pixel)
    print('median width (cm):', profiles['width_median'] * cm_per_pixel)


def main():
    # File paths
    file_list = [
        'tmp/long_narrow_stem/warped_8/2.png',
        # 'long_narrow_stem/warped_8/2.png',
        # 'long_narrow_stem/warped_8/3.png',
        # 'short_thick_stem/warped_1/1.png',
        # 'short_thick_stem/warped_1/2.png',
        # 'short_thick_stem/warped_1/3.png',
        # 'no_stem/warped_11/0.png',
        # 'no_stem/warped_11/1.png',
        # 'no_stem/warped_11/2.png',
        # 'no_stem/warped_11/3.png',
        # 'no_stem/warped_11/4.png',
        # 'no_stem/warped_17/0.png',
        # 'no_stem/warped_17/1.png',
        # 'no_stem/warped_17/3.png',
        # 'no_stem/warped_17/4.png',
        # 'no_stem/warped_17/5.png'
        # "test_set/small_objects/warped_51/1.png",
        # "test_set/small_objects/warped_30/5.png",
        # "test_set/small_objects/warped_201B/3.png",
    ]

    for f in file_list:
        print('Processing mask file at:', f)
        visualize(f)
        print('############################\n')


if __name__ == '__main__':
    main()
