
This is demo code to compute sparse code gradients for high-quality contour
detection in natural images, developed by Xiaofeng Ren at Intel Labs.
This software is released under the BSD license; see COPYRIGHT.txt. The current
version is v1.1.1.

It currently supports contour detection on color, depth, or RGB-D
(color+depth) channels, using models pre-trained on BSDS500 and NYU Depth (v2).
Tested on Ubuntu 12.04 64-bit.

The matlab code has three external dependencies:

  1. Orthogonal Matching Pursuit code from Ron Rubinstein, under ./omp
     (see http://www.cs.technion.ac.il/~ronrubin/software.html)

  2. Global Pb code from Berkeley, under ./gpb
     (see http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

     Only a subset of the functions are needed, for smoothing/nonmax
     suppression, and optionally globalization.

  3. The matrix template library Eigen, under ./Eigen-3.1
     (see http://eigen.tuxfamily.org/index.php?title=Main_Page)

Run compile.m to compile the SCG mex code as well as the OMP code (needs to be
changed if you use Windows). To recompile the functions under ./gpb, please
download the full gPb software and follow instructions there.

To compute the local version of sparse code gradients,

  scg_local=scg_pb_orient_Lab(img,scg_model_Lab);

To compute the global version of sparse code gradients using the gPb globalization,

  [scg_local,scg_global]=scg_pb_orient_Lab(img,scg_model_Lab);

Use script_run_BSDS500_test.m to run the code on the BSDS500 benchmark.
./scg_global_BSDS already contains the global SCG outputs on the BSDS500 test and
its benchmarking results. 

I also include similar scripts for the NYU Depth (v2) dataset. In ./nyu_v2, you
can find more information and a script convert_dataset.m to download the NYU
Depth dataset (v1) and convert it to the BSDS format with half resolution.

The resulting RGBD contours are not packaged here for space reasons, but I do
include boundary benchmarking numbers using standard BSDS setting.

The current implementation consumes about ~3.5G memory for both BSDS images
(color SCG) and NYU depth images (RGBD SCG).

For more information, please refer to our NIPS 2012 paper:

Discriminatively Trained Sparse Code Gradients for Contour Detection
Xiaofeng Ren and Liefeng Bo
Advances in Neural Information Processing Systems (NIPS), 2012.

I will not be able to provide technical support for this code. Nonetheless,
Let me know (xren@cs.washington.edu) if you have any comments and suggestions.


