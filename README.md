# muxable/slam

Simultaneous Localization and Mapping.

## Algorithm

The algorithm follows the following steps:

1. For each image, compute keypoints with ORB (https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html)
2. Match keypoints with previous image
3. Compute essential matrix from matched keypoints (https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html)
4. Compute motion (rotation and transformation) from essential matrix
5. Store relative motion for each frame
6. Identify nearby points with g2o
7. Correct accumulated errors
