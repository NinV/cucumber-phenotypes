from typing import Optional
import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis
from scipy.interpolate import splprep, splev
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon

from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import networkx as nx


def calc_skeleton_line(binary_mask, method='zhang'):
    if method == 'medial_axis':
        skeleton = medial_axis(binary_mask, return_distance=False)
    elif method == 'zhang':
        skeleton = skeletonize(binary_mask, method='zhang')
    elif method == 'lee':
        skeleton = skeletonize(binary_mask, method='lee')
    else:
        raise ValueError('Method must be in {"zhang", "lee", "medial_axis"}')

    coords = np.argwhere(skeleton)[:, ::-1]  # coords = (n, 2) = [[x0, y0], [x1, y1], ...]
    return skeleton, coords


def curve_fitting(coords, k=3, s=None):
    if s is None:
        s = coords.shape[0]
    tck, _ = splprep(coords.T, k=k, s=s)
    return tck


def build_polygon(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    points = [tuple(point[0]) for point in largest_contour]
    polygon = Polygon(points)
    return polygon


def chord_length_parameterization_method(tck, num_samples=1000):
    # Compute the chord lengths
    t = np.linspace(0, 1, num_samples)
    curve_points = np.array(splev(t, tck)).T
    chord_lengths = np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)

    # Compute the cumulative sum of chord lengths
    # cumulative_lengths = np.cumsum(chord_lengths)
    cumulative_lengths = np.concatenate(([0], np.cumsum(chord_lengths)))
    total_length = cumulative_lengths[-1]

    # Compute the parameter values corresponding to the desired number of samples
    parameter_values = np.interp(np.linspace(0, total_length, num_samples), cumulative_lengths, t)

    # Evaluate the spline at the computed parameter values
    curve_points = np.array(splev(parameter_values, tck)).T

    # Calculate the length of the curve
    curve_length = np.sum(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1))
    return curve_points, parameter_values, curve_length


def arc_length_parameterization_method(tck, num_samples=1000):
    # Evaluate the spline at equidistant parameter values
    t = np.linspace(0, 1, num_samples)
    curve_points = np.array(splev(t, tck)).T
    # Calculate the length of the curve
    curve_length = np.sum(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1))
    return curve_points, t, curve_length


def cal_spline_length(tck, method='chord_length', **kwargs):
    if method == 'chord_length':
        return chord_length_parameterization_method(tck, **kwargs)
    elif method == 'arc_length':
        return arc_length_parameterization_method(tck, **kwargs)
    else:
        raise ValueError('Not recognize method="{}"'.format(method))


def derivative(tck, t, h=1e-5):
    p1, p2 = np.array(splev([t - h, t + h], tck)).T
    return (p2 - p1) / (2 * h)


def get_intersection_coordinates(geometry):
    if isinstance(geometry, LineString) and len(geometry.coords) == 2:
        return geometry
    raise ValueError


class CucumberShape:
    def __init__(self, skel_method='zhang', bspline_k=3, bspline_s='auto', bspline_num_points=1000):
        self.skel_method = skel_method
        self.bspline_k = bspline_k
        self.bspline_s = bspline_s
        self.bspline_num_points = bspline_num_points

    def find_medial_axis(self, binary_mask):
        skeleton, coords = calc_skeleton_line(binary_mask, self.skel_method)

        if self.bspline_s == 'auto':
            skel_tck = curve_fitting(coords, k=self.bspline_k)
        else:
            skel_tck = curve_fitting(coords, k=self.bspline_k, s=self.bspline_s)
        boundary = build_polygon(binary_mask)

        # extending the bspline curve to find intersection
        t = np.linspace(-0.1, 1.1, self.bspline_num_points)
        curve_points = splev(t, skel_tck)
        curve_points = np.array([(x, y) for (x, y) in zip(*curve_points)])
        curve_line = LineString(coordinates=curve_points)
        p1, p2 = curve_line.intersection(boundary).boundary.geoms
        intersections = np.array([[p1.x, p1.y], [p2.x, p2.y]])
        intersect_cdist = cdist(curve_points, intersections)
        p1_idx, p2_idx = np.argmin(intersect_cdist, axis=0)
        curve_points[p1_idx] = p1.x, p1.y
        curve_points[p2_idx] = p2.x, p2.y
        curve_points = curve_points[min(p1_idx, p2_idx): max(p1_idx, p2_idx) + 1]
        curve_tck = curve_fitting(curve_points, k=self.bspline_k)
        return curve_tck, skeleton, coords, boundary

    def width_profile(self, tck, boundary: Polygon, num_samples=100, parameterized_method='chord_length',
                      peak_thresh=0.3, max_stem_width=0.2):
        # boundary size
        x_min, y_min, x_max, y_max = boundary.bounds
        max_size = (x_max - x_min, y_max - y_min)

        # b-spline curve_points and normal_vectors
        curve_points, t, curve_length = cal_spline_length(tck, method=parameterized_method, num_samples=num_samples)
        dx, dy = splev(t, tck, der=1)  # derivatives
        normal_vectors = np.stack((-dy, dx), axis=1)
        normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=1, keepdims=True)

        # constructing normal line
        p1s = curve_points + normal_vectors * max_size
        p2s = curve_points - normal_vectors * max_size

        normal_lines = []
        indices = []
        widths = []
        for i, (p1, p2) in enumerate(zip(p1s, p2s)):
            line = LineString([p1, p2])
            intersections = line.intersection(boundary)
            try:
                width_line_segment = get_intersection_coordinates(intersections)
            except ValueError:
                continue
            # p1, p2 = coords
            normal_lines.append(width_line_segment)
            widths.append(normal_lines[-1].length)
            indices.append(i)

        widths = np.array(widths)
        width_median = np.median(widths)

        if len(indices) != curve_points.shape[0]:
            curve_points = curve_points[indices]
            normal_vectors = normal_vectors[indices]
            t = t[indices]

        chord_lengths = np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)
        # curve_length = np.sum(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1))
        cumulative_lengths = np.concatenate(([0], np.cumsum(chord_lengths)))
        abs_change_rate = np.zeros(widths.shape[0])
        abs_change_rate[1:] = np.abs(widths[1:] - widths[0:-1])

        # finding stem region
        # assume stem is from first half (important)
        n = widths.shape[0]
        peak_idx = np.argmax(abs_change_rate[:n//2])
        if abs_change_rate[peak_idx] > peak_thresh * width_median:      # threshold test for long and narrow stem
            stem_idx = peak_idx
        else:                                                           # short and thick stem
            # find the first idx where width less than max_stem_width
            stem_idx = np.argmin(widths[:n//2] < width_median * max_stem_width)
        # curve_length_wo_stem = cumulative_lengths[-1] - cumulative_lengths[stem_idx]
        curve_length_wo_stem = curve_length - cumulative_lengths[stem_idx]
        profiles = {'curve_points': curve_points,
                    'normal_vectors': normal_vectors,
                    'parameter_values': t,
                    'curve_length': curve_length,
                    'tck': tck,
                    'method': parameterized_method,
                    'normal_lines': normal_lines,
                    'widths': widths,
                    'indices': indices,
                    'width_median': width_median,
                    'abs_peak_threshold': peak_thresh * width_median,
                    'cumulative_lengths': cumulative_lengths,
                    'abs_change_rate': abs_change_rate,
                    'peak_idx': peak_idx,
                    'curve_length (remove stem)': curve_length_wo_stem,
                    'stem_idx': stem_idx
                    }
        return profiles

    @staticmethod
    def plot_chart(profile, save_loc='plot.png', conversion_scale=1, unit='pixels'):
        max_y = np.max(profile['widths']) * conversion_scale
        max_x = np.max(profile['cumulative_lengths']) * conversion_scale
        plt.plot(profile['cumulative_lengths'] * conversion_scale,
                 profile['widths'] * conversion_scale,
                 label='width', color='#1f77b4')
        plt.plot(profile['cumulative_lengths'] * conversion_scale,
                 profile['abs_change_rate'] * conversion_scale,
                 label='abs. change rate', color='#ff7f0e')
        plt.axhline(y=profile['width_median'] * conversion_scale, color='magenta', linestyle='--', label="width median")
        plt.text(x=max_x / 2,
                 y=profile['width_median'] * conversion_scale + 0.02 * max_y,
                 s=round(profile['width_median'] * conversion_scale, 2),
                 color='magenta')

        # peak thresholding
        plt.axhline(y=profile['abs_peak_threshold'] * conversion_scale, color='r', linestyle='--', label="peak thresh.")
        plt.text(x=max_x / 2,
                 y=profile['abs_peak_threshold'] * conversion_scale + 0.02 * max_y,
                 s=round(profile['abs_peak_threshold'] * conversion_scale, 2),
                 color='r')
        peak_idx = profile['peak_idx']
        stem_idx = profile['stem_idx']
        plt.axvline(x=profile['cumulative_lengths'][peak_idx] * conversion_scale,
                    color='gray', linestyle='--')
        plt.axhline(y=profile['abs_change_rate'][peak_idx] * conversion_scale, color='#ff7f0e', linestyle='--')
        plt.plot(profile['cumulative_lengths'][peak_idx] * conversion_scale,
                 profile['abs_change_rate'][peak_idx] * conversion_scale, '^', color='#ff7f0e',
                 label='peak change rate')
        plt.text(
                 # x=profile['cumulative_lengths'][idx] * conversion_scale,
                 x=max_x / 2,
                 y=profile['abs_change_rate'][peak_idx] * conversion_scale + 0.02 * max_y,
                 s=round(profile['abs_change_rate'][peak_idx] * conversion_scale, 2),
                 color='#ff7f0e')

        # plt.axhline(y=profile['widths'][stem_idx] * conversion_scale, color='#1f77b4', linestyle='--')
        plt.plot(profile['cumulative_lengths'][stem_idx] * conversion_scale,
                 profile['widths'][stem_idx] * conversion_scale, 'o', color='#1f77b4',
                 label='stem point')
        plt.text(
            # x=max_x / 2,
            x=profile['cumulative_lengths'][stem_idx] * conversion_scale + 0.02 * max_x,
            # y=profile['widths'][stem_idx] * conversion_scale + 0.02 * max_y,
            y=profile['widths'][stem_idx] * conversion_scale,
            s=(round(profile['cumulative_lengths'][stem_idx] * conversion_scale, 2), round(profile['widths'][stem_idx] * conversion_scale, 2)),
            color='#1f77b4')

        plt.xlabel('cumulative length ({})'.format(unit))
        plt.ylabel('{}'.format(unit))
        plt.tight_layout(pad=0)
        plt.legend(bbox_to_anchor=(0.8, -0.1), ncol=2, fancybox=True, shadow=True)
        plt.xlim(0, max_x)
        plt.ylim(0, max_y)
        plt.savefig(save_loc, bbox_inches="tight")
        plt.clf()
