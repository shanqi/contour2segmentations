%
%  Xiaofeng Ren <xiaofeng.ren@intel.com>, 07/2012
%

function [pb_ori]=scg_gradient_Lab( omp_codes_L, omp_codes_ab, omp_codes_depth, omp_codes_normal, patchsize_L, patchsize_ab, patchsize_depth, patchsize_normal, scales_L, scales_ab, scales_depth, scales_normal, ori, model, beta, discratio, nori )
%
% function [pb_ori]=scg_gradient_Lab( omp_codes_L, omp_codes_ab, omp_codes_depth, omp_codes_normal, patchsize_L, patchsize_ab, patchsize_deph, patchsize_normal, scales_L, scales_ab, scales_depth, scales_normal, ori, model, beta, discratio, nori )
%
%

assert(length(beta)==2);

assert(patchsize_L==patchsize_ab); assert(patchsize_L==patchsize_depth); assert(patchsize_L==patchsize_normal);

omp_codes_L=single(omp_codes_L);
omp_codes_ab=single(omp_codes_ab*beta(1));
omp_codes_depth=single(omp_codes_depth);
omp_codes_normal=single(omp_codes_normal*beta(2));

model=reshape(model,[length(model) 1]);

nscale_L=length(scales_L);
nscale_ab=length(scales_ab);
nscale_depth=length(scales_depth);
nscale_normal=length(scales_normal);
nscale=nscale_L+nscale_ab+nscale_depth+nscale_normal;

ndict_L=size(omp_codes_L,3);
ndict_ab=size(omp_codes_ab,3);
ndict_depth=size(omp_codes_depth,3);
ndict_normal=size(omp_codes_normal,3);

assert( length(model)==(nscale_L*ndict_L+nscale_ab*ndict_ab+nscale_depth*ndict_depth+nscale_normal*ndict_normal)*4 );

margin=max( [max(scales_L) max(scales_ab) max(scales_depth) max(scales_normal)] );

[h0,w0,dmmy]=size(omp_codes_L);
h0=h0+(patchsize_L-1); w0=w0+(patchsize_L-1);

omp_codes_L=padarray(omp_codes_L,[floor((patchsize_L-1)/2) patchsize_L-1-floor((patchsize_L-1)/2) 0]);
omp_codes_L=padarray(omp_codes_L,[margin margin 0]);
omp_codes_L=imrotate( omp_codes_L, -(ori-1)/nori*180 );
omp_codes_ab=padarray(omp_codes_ab,[floor((patchsize_ab-1)/2) patchsize_ab-1-floor((patchsize_ab-1)/2) 0]);
omp_codes_ab=padarray(omp_codes_ab,[margin margin 0]);
omp_codes_ab=imrotate( omp_codes_ab, -(ori-1)/nori*180 );
omp_codes_depth=padarray(omp_codes_depth,[floor((patchsize_depth-1)/2) patchsize_depth-1-floor((patchsize_depth-1)/2) 0]);
omp_codes_depth=padarray(omp_codes_depth,[margin margin 0]);
omp_codes_depth=imrotate( omp_codes_depth, -(ori-1)/nori*180 );
omp_codes_normal=padarray(omp_codes_normal,[floor((patchsize_normal-1)/2) patchsize_normal-1-floor((patchsize_normal-1)/2) 0]);
omp_codes_normal=padarray(omp_codes_normal,[margin margin 0]);
omp_codes_normal=imrotate( omp_codes_normal, -(ori-1)/nori*180 );

%----------------start------------------
pb_ori=compute_linear_gradient_RGBD_mex( omp_codes_L, scales_L, omp_codes_ab, scales_ab, omp_codes_depth, scales_depth, omp_codes_normal, scales_normal, model, discratio );
%_----------------end-------------------

pb_ori=imrotate( pb_ori, (ori-1)/nori*180 );
[h2,w2]=size(pb_ori);

imcenter=round([h2 w2]/2);
imhalfsize0=floor([h0 w0]/2);

pb_ori=pb_ori( imcenter(1)-imhalfsize0(1):imcenter(1)-imhalfsize0(1)+h0-1 , imcenter(2)-imhalfsize0(2):imcenter(2)-imhalfsize0(2)+w0-1 );
pb_ori=double(pb_ori);


