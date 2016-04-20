%
%  Xiaofeng Ren <xiaofeng.ren@intel.com>, 07/2012
%

function [pb_ori]=scg_gradient_Lab( omp_codes_L, omp_codes_ab, patchsize_L, patchsize_ab, scales_L, scales_ab, ori, model, beta, discratio, nori )
%
%function [pb_ori]=scg_gradient_Lab( omp_codes_L, omp_codes_ab, patchsize_L, patchsize_ab, scales_L, scales_ab, ori, model, beta, discratio, nori )
%
%

omp_codes_L=single(omp_codes_L);
omp_codes_ab=single(omp_codes_ab*beta);

model=reshape(model,[length(model) 1]);
nscale=length(scales_L)+length(scales_ab);

ndict=size(omp_codes_L,3);
assert( length(model)==nscale*ndict*4 );

margin=max( max(scales_L), max(scales_ab) );

[h0,w0,~]=size(omp_codes_L);
h0=h0+(patchsize_L-1); w0=w0+(patchsize_L-1);

omp_codes_L=padarray(omp_codes_L,[floor((patchsize_L-1)/2) patchsize_L-1-floor((patchsize_L-1)/2) 0]);
omp_codes_L=padarray(omp_codes_L,[margin margin 0]);
omp_codes_L=imrotate( omp_codes_L, -(ori-1)/nori*180 );
omp_codes_ab=padarray(omp_codes_ab,[floor((patchsize_ab-1)/2) patchsize_ab-1-floor((patchsize_ab-1)/2) 0]);
omp_codes_ab=padarray(omp_codes_ab,[margin margin 0]);
omp_codes_ab=imrotate( omp_codes_ab, -(ori-1)/nori*180 );

%----------------start------------------
pb_ori=compute_linear_gradient_Lab_mex( omp_codes_L, scales_L, omp_codes_ab, scales_ab, model, discratio );
%_----------------end-------------------

pb_ori=imrotate( pb_ori, (ori-1)/nori*180 );
[h2,w2]=size(pb_ori);

imcenter=round([h2 w2]/2);
imhalfsize0=floor([h0 w0]/2);

pb_ori=pb_ori( imcenter(1)-imhalfsize0(1):imcenter(1)-imhalfsize0(1)+h0-1 , imcenter(2)-imhalfsize0(2):imcenter(2)-imhalfsize0(2)+w0-1 );
pb_ori=double(pb_ori);


