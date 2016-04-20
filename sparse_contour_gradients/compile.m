cd omp/private
make
cd ../..
mex -O compute_linear_gradient_Lab_mex.cc -I./Eigen-3.1/ CXXFLAGS="\$CXXFLAGS -O3 -march=native -mfpmath=sse -DNDEBUG"
mex -O compute_linear_gradient_RGBD_mex.cc -I./Eigen-3.1/ CXXFLAGS="\$CXXFLAGS -O3 -march=native -mfpmath=sse -DNDEBUG"
