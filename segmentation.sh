#!/bin/bash

# download SAM weights
# mkdir sam_weights
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# mv sam_vit_b_01ec64.pth sam_weights/
python sam.py -i data/warped/ -o data/segmentation_results --mode points