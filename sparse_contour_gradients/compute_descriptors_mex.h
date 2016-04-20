//
//  Xiaofeng Ren <xiaofeng.ren@intel.com>, 07/2012
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <iostream>
#include <numeric>

#include "mex.h"

using namespace std;
using namespace Eigen;

const float eps=1e-6;

void cumsum(MatrixXf& x)
{
   // cumsum, column-wise (default order)
   for(int i=0; i<x.cols(); i++) {
      float* d=x.data()+i*x.rows();
      partial_sum(d,d+x.rows(),d);
   }
   return;
}

void double_cumsum(MatrixXf& x)
{
   MatrixXf x2;
   cumsum(x);
   x2=x.transpose();
   cumsum(x2);
   x=x2.transpose();
   return;
}



void compute_descriptor(const float* omp_codes_input, const double* scales, MatrixXf& descL, MatrixXf& descR, int height, int width, int ndict, int nscale, double discratio)
{
    assert( descL.rows()*descL.cols()==height*width*ndict*nscale );
    assert( descR.rows()*descR.cols()==height*width*ndict*nscale );

    float* descL_data=descL.data();
    float* descR_data=descR.data();

    MatrixXf omp_int(height,width);
    MatrixXf omp_int_c(height,width);
    MatrixXf omp_max(height,width);
    MatrixXf desc(height,width);

    //#pragma omp parallel for
    for(int idict=0; idict<ndict; idict++) {

       memcpy( omp_int.data(), omp_codes_input+(idict*height*width),height*width*sizeof(float));
       omp_int_c = ( omp_int.array()>eps ).cast<float>();

       double_cumsum( omp_int );
       double_cumsum( omp_int_c );

       for(int iscale=0; iscale<nscale; iscale++) {
          int s=(int)(scales[iscale]*discratio+0.5);
          int s2_half=scales[iscale];
          int s2=s2_half*2+1;
          MatrixXf a = omp_int.block(s2,s,height-s2-1,width-s-1)-omp_int.block(0,s,height-s2-1,width-s-1)-omp_int.block(s2,0,height-s2-1,width-s-1)+omp_int.block(0,0,height-s2-1,width-s-1);
          MatrixXf c = omp_int_c.block(s2,s,height-s2-1,width-s-1)-omp_int_c.block(0,s,height-s2-1,width-s-1)-omp_int_c.block(s2,0,height-s2-1,width-s-1)+omp_int_c.block(0,0,height-s2-1,width-s-1);
          omp_max.setZero();
          omp_max.block(1,1,height-s2-1,width-s-1) = a.array() / ( c.array() + (c.array()<=eps).cast<float>() );
          desc.setZero();
          desc.block(s2_half,s,height-s2-1,width-s-2) = omp_max.block(0,0,height-s2-1,width-s-2).array().max( ArrayXf::Zero(height,width) );
          memcpy(descL_data+(idict*height*width)+(iscale*ndict*height*width),desc.data(),height*width*sizeof(float));
          desc.setZero();
          desc.block(s2_half,s,height-s2-1,width-s-2) = omp_max.block(0,s+1,height-s2-1,width-s-2).array().max( ArrayXf::Zero(height,width) );
          memcpy(descR_data+(idict*height*width)+(iscale*ndict*height*width),desc.data(),height*width*sizeof(float));
       }

    }

    return;
}



