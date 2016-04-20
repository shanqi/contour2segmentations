%
%  Xiaofeng Ren <xiaofeng.ren@intel.com>, 07/2012
%

function omp_codes=compute_omp_codes( im, dic )
%
% function omp_codes=compute_omp_codes( im, dic )
%
%

X = im2col_multi(im, [dic.patchsize dic.patchsize], 'sliding', size(im,3));
X = remove_dc(X, 'columns');
omp_codes = omp(dic.dic'*X, dic.dic'*dic.dic, dic.sparsity, 'gammamode', 'full');
omp_codes = abs(reshape(full(omp_codes'), size(im,1)-dic.patchsize+1,size(im,2)-dic.patchsize+1, dic.dicsize));

