import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
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

    coords = np.argwhere(skeleton)
    return skeleton, coords


def graph_based_curve_length(coords):
    # Create a graph and add nodes
    graph = nx.Graph()
    graph.add_nodes_from(range(len(coords)))

    # Connect neighboring points based on smallest distance
    dist_matrix = cdist(coords, coords)
    neighborhood_radius = 3  # Adjust this parameter to define the neighborhood size
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = dist_matrix[i, j]
            if dist <= neighborhood_radius:
                graph.add_edge(i, j, weight=dist)
    # Find the shortest path between two points
    shortest_path = nx.shortest_path(graph, source=0, target=len(coords) - 1)

    # Calculate the length of the shortest path in terms of pixels and Euclidean distance
    shortest_path_length = sum(graph[u][v]['weight'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    return shortest_path_length


def curve_fitting(coords):
    s_values = np.linspace(0, 1, num=10)

    # Calculate the average error for each 's' value
    errors = []
    params = []
    for s in s_values:
        # Perform spline fitting to get a smooth curve approximation
        tck, _ = splprep(coords.T, s=s)
        params.append(tck)

        # Evaluate the spline at the original coordinates
        curve_points = np.array(splev(coords[:, 0], tck)).T

        # Calculate the mean squared error
        mse = np.mean(np.square(coords - curve_points))
        errors.append(mse)

    # Find the index of the minimum error
    best_index = np.argmin(errors)
    best_s = s_values[best_index]
    best_error = errors[best_index]
    best_params = params[best_index]

    return best_s, best_error, best_params


def arc_length_parameterization_method(coords, tck=None, num_samples=1000):
    if tck is None:
        _, _, tck = curve_fitting(coords)

    # Evaluate the spline at equidistant parameter values
    t = np.linspace(0, 1, num_samples)
    curve_points = np.array(splev(t, tck)).T
    # Calculate the length of the curve
    curve_length = np.sum(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1))
    return curve_length


def chord_length_parameterization_method(coords, tck=None, num_samples=1000):
    if tck is None:
        _, _, tck = curve_fitting(coords)

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
    return curve_length


def main():
    # Assume binary_mask is the binary mask representing the shape
    binary_mask = cv2.imread('0.png', cv2.IMREAD_GRAYSCALE)

    # Compute the skeleton
    skeleton, coords = calc_skeleton_line(binary_mask)

    curve_length_graph = graph_based_curve_length(coords)

    _, _, tck = curve_fitting(coords)
    curve_length_arc = arc_length_parameterization_method(coords, tck, 1000)
    curve_length_chord = chord_length_parameterization_method(coords, tck, 1000)

    print("Curve length (Graph based method):", curve_length_graph)
    print("Curve length (arc method):", curve_length_arc)
    print("Curve length (chord method):", curve_length_chord)


if __name__ == '__main__':
    main()

