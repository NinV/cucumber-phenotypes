#!/bin/bash
python evaluate_length_measurement.py -i data/segmentation_results -l data/test_set/labels_test_91.csv -o "data/measurements.json" --peak_thresh 0.2 --max_stem_width 0.4
echo "Write result to data/measurements.json"
echo -e "#####################################################################\n"


