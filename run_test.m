%% run_test.m

addpath( './sparse_contour_gradients/');
addpath( './sparse_contour_gradients/omp/' );
addpath( './sparse_contour_gradients/gpb/' );
addpath( './BSR/grouping/' )
addpath( './BSR/grouping/lib' )

contour2segmentation('./images/IMG_0011.JPGGpbAll.ary');

