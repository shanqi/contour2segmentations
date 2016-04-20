//
//  Xiaofeng Ren <xiaofeng.ren@intel.com>, 07/2012
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <iostream>
#include <numeric>
#include <time.h>

#include "mex.h"

#include "compute_descriptors_mex.h"

using namespace std;
using namespace Eigen;

VectorXf rowwise_norm(const MatrixXf& x)
{
  VectorXf d=VectorXf::Zero(x.rows(),1);
  for(int i=0; i<x.cols(); i++) {
    //d.noalias()+=(x.col(i).array() * x.col(i).array());
    d.array()+=(x.col(i).array().square());
  }
  d = d.array().sqrt();
  return d;
}

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    clock_t time_start, time_now;

    time_start=clock();

    //Eigen::setNbThreads(4);

    if ((nrhs!=10) || (nlhs!=1)) {
      mexErrMsgTxt("Usage : [pb_ori] = compute_linear_gradient_Lab_mex( omp_codes_L, scales_L, omp_codes_ab, scales_ab, omp_codes_depth, scales_depth, omp_codes_normal, scales_normal, model, discratio );");
    }

    const mxArray* omp_codes_L_mat = prhs[0];
    const mxArray* scales_L_mat = prhs[1];
    int nscale_L=(mxGetM(scales_L_mat)*mxGetN(scales_L_mat));
    double* scales_L=mxGetPr(scales_L_mat);
    const mxArray* omp_codes_ab_mat = prhs[2];
    const mxArray* scales_ab_mat = prhs[3];
    int nscale_ab=(mxGetM(scales_ab_mat)*mxGetN(scales_ab_mat));
    double* scales_ab=mxGetPr(scales_ab_mat);
    const mxArray* omp_codes_depth_mat = prhs[4];
    const mxArray* scales_depth_mat = prhs[5];
    int nscale_depth=(mxGetM(scales_depth_mat)*mxGetN(scales_depth_mat));
    double* scales_depth=mxGetPr(scales_depth_mat);
    const mxArray* omp_codes_normal_mat = prhs[6];
    const mxArray* scales_normal_mat = prhs[7];
    int nscale_normal=(mxGetM(scales_normal_mat)*mxGetN(scales_normal_mat));
    double* scales_normal=mxGetPr(scales_normal_mat);

    const mxArray* model_mat = prhs[8];
    double discratio=mxGetScalar(prhs[9]);

    // omp_codes must be single precision
    if( !mxIsSingle(omp_codes_L_mat) ) {
      mexPrintf("Error: omp_codes must be single precision\n"); return;
    }
    if( !mxIsSingle(omp_codes_ab_mat) ) {
      mexPrintf("Error: omp_codes must be single precision\n"); return;
    }
    if( !mxIsSingle(omp_codes_depth_mat) ) {
      mexPrintf("Error: omp_codes must be single precision\n"); return;
    }
    if( !mxIsSingle(omp_codes_normal_mat) ) {
      mexPrintf("Error: omp_codes must be single precision\n"); return;
    }

    // scales must be double precision
    if( !mxIsDouble(scales_ab_mat) ) {
      mexPrintf("Error: scales must be double precision\n"); return;
    }
    if( !mxIsDouble(scales_ab_mat) ) {
      mexPrintf("Error: scales must be double precision\n"); return;
    }
    if( !mxIsDouble(scales_depth_mat) ) {
      mexPrintf("Error: scales must be double precision\n"); return;
    }
    if( !mxIsDouble(scales_normal_mat) ) {
      mexPrintf("Error: scales must be double precision\n"); return;
    }

    // model must be double
    if( !mxIsDouble(model_mat) ) {
      mexPrintf("Error: model must be double precision\n"); return;
    }

    assert( mxGetNumberOfDimensions(omp_codes_L_mat)==3 );
    const mwSize* sizes_L = mxGetDimensions(omp_codes_L_mat);
    mwSize height=sizes_L[0];
    mwSize width=sizes_L[1];
    mwSize ndict_L=sizes_L[2];

    assert( mxGetNumberOfDimensions(omp_codes_ab_mat)==3 );
    const mwSize* sizes_ab = mxGetDimensions(omp_codes_ab_mat);
    assert( height==size_ab[0] );
    assert( width==size_ab[1] );
    mwSize ndict_ab=sizes_ab[2];

    assert( mxGetNumberOfDimensions(omp_codes_depth_mat)==3 );
    const mwSize* sizes_depth = mxGetDimensions(omp_codes_depth_mat);
    assert( height==size_depth[0] );
    assert( width==size_depth[1] );
    mwSize ndict_depth=sizes_depth[2];

    assert( mxGetNumberOfDimensions(omp_codes_normal_mat)==3 );
    const mwSize* sizes_normal = mxGetDimensions(omp_codes_normal_mat);
    assert( height==size_normal[0] );
    assert( width==size_normal[1] );
    mwSize ndict_normal=sizes_normal[2];

    plhs[0] = mxCreateDoubleMatrix(height,width,mxREAL);

    int npixel=height*width;
    int nfeat_L=ndict_L*nscale_L;
    int nfeat_ab=ndict_ab*nscale_ab;
    int nfeat_depth=ndict_depth*nscale_depth;
    int nfeat_normal=ndict_depth*nscale_normal;
    int nfeat=nfeat_L+nfeat_ab+nfeat_depth+nfeat_normal;

    MatrixXf descL_L( npixel, nfeat_L );
    MatrixXf descR_L( npixel, nfeat_L );
    compute_descriptor( (float*)mxGetData(omp_codes_L_mat), scales_L, descL_L, descR_L, height, width, ndict_L, nscale_L, discratio );

    MatrixXf descL_ab( npixel, nfeat_ab );
    MatrixXf descR_ab( npixel, nfeat_ab );
    compute_descriptor( (float*)mxGetData(omp_codes_ab_mat), scales_ab, descL_ab, descR_ab, height, width, ndict_ab, nscale_ab, discratio );

    MatrixXf descL_depth( npixel, nfeat_depth );
    MatrixXf descR_depth( npixel, nfeat_depth );
    compute_descriptor( (float*)mxGetData(omp_codes_depth_mat), scales_depth, descL_depth, descR_depth, height, width, ndict_depth, nscale_depth, discratio );

    MatrixXf descL_normal( npixel, nfeat_normal );
    MatrixXf descR_normal( npixel, nfeat_normal );
    compute_descriptor( (float*)mxGetData(omp_codes_normal_mat), scales_normal, descL_normal, descR_normal, height, width, ndict_normal, nscale_normal, discratio );

    //time_now=clock(); cout << "descriptors done: " << (double)(time_now-time_start) / ((double)CLOCKS_PER_SEC) << endl;

    MatrixXf descL( npixel, nfeat );
    descL << descL_L, descL_ab, descL_depth, descL_normal;

    descL_L.resize(0,0); descL_ab.resize(0,0); descL_depth.resize(0,0); descL_normal.resize(0,0);

    MatrixXf descR( npixel, nfeat );
    descR << descR_L, descR_ab, descR_depth, descR_normal;
    descR_L.resize(0,0); descR_ab.resize(0,0); descR_depth.resize(0,0); descR_normal.resize(0,0);

    VectorXf dL=rowwise_norm( descL );
    VectorXf dR=rowwise_norm( descR );

    VectorXf d=(dL+dR).array()/2+1;

    descL.array().colwise() /= d.array();
    descR.array().colwise() /= d.array();

    //time_now=clock(); cout << "normalization done: " << (double)(time_now-time_start) / ((double)CLOCKS_PER_SEC) << endl;

    MatrixXf desc_m=(descL-descR).array().abs();
    MatrixXf desc_p=(descL+descR).array().abs();

    //time_now=clock(); cout << "desc_m desc_p done: " << (double)(time_now-time_start) / ((double)CLOCKS_PER_SEC) << endl;

    descL.resize(0,0); descR.resize(0,0);

    MatrixXf m1=desc_m.array().sqrt();
    m1.resize(npixel,nfeat);
          desc_m.resize(0,0);
    MatrixXf m2=m1.array().sqrt();
    m2.resize(npixel,nfeat);

    //time_now=clock(); cout << "sqrt m1 m2 done: " << (double)(time_now-time_start) / ((double)CLOCKS_PER_SEC) << endl;

    MatrixXf model = Map<MatrixXd>( mxGetPr(model_mat),nfeat*4,1 ).cast<float>();

    MatrixXf pb = m2 * model.block(0,0,nfeat,1).col(0);
    pb += ( m1.array() * m2.array() ).matrix() * model.block(nfeat*2,0,nfeat,1).col(0);
          m2.resize(0,0);
          m1.resize(0,0);

    //time_now=clock(); cout << "model 1 done: " << (double)(time_now-time_start) / ((double)CLOCKS_PER_SEC) << endl;

    MatrixXf p1=desc_p.array().sqrt();
          desc_p.resize(0,0);
    MatrixXf p2=p1.array().sqrt();

    //time_now=clock(); cout << "sqrt p1 p2 done: " << (double)(time_now-time_start) / ((double)CLOCKS_PER_SEC) << endl;

    pb += ( p2 * model.block(nfeat*1,0,nfeat,1).col(0) );
    pb += ( p1.array() * p2.array() ).matrix() * model.block(nfeat*3,0,nfeat,1).col(0);
          p2.resize(0,0);
          p1.resize(0,0);

    //time_now=clock(); cout << "all done: " << (double)(time_now-time_start) / ((double)CLOCKS_PER_SEC) << endl;

    MatrixXd pb_d = pb.cast<double>();
    memcpy( mxGetPr(plhs[0]), pb_d.data(), height*width*sizeof(double) );

    return;
}







