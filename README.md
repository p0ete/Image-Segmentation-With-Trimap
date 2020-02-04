# Image Segmentation With Trimap

This repository is another version of the repository **Segmentation based Semantic Matting**
 (https://github.com/Griffin98/Segmentation_based_Semantic_Matting) based on the paper  **Instance Segmentation based Semantic Matting for Compositing Applications** (https://arxiv.org/abs/1904.05457).

### Run
1. Run the setup file in order to download the two models:

> ./setup.sh

2. Place your input images in the Data/input/ directory.

3. Run:

> python -W ignore demo_end_to_end.py

4. Find the foreground image in the Data/foreground/ directory. 
