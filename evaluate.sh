#!/bin/bash
for PEAK_THRESH in 0.0 0.1 0.2 0.3 0.4 0.5
do
  for MAX_STEM_WIDTH in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
  do
    echo "peak_thresh=${PEAK_THRESH}, max_stem_width=${MAX_STEM_WIDTH}."
    python evaluate_length_measurement.py -i test_set/test_91_sort_coords -l test_set/labels_test_91.csv -o "out_peak-${PEAK_THRESH}_stem-${MAX_STEM_WIDTH}.json" --peak_thresh "${PEAK_THRESH}" --max_stem_width "${MAX_STEM_WIDTH}"
    echo "Write result to out_peak-${PEAK_THRESH}_stem-${MAX_STEM_WIDTH}.json"
    echo -e "#####################################################################\n"
  done
done

