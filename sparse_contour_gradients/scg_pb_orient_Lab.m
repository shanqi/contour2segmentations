%
%  Xiaofeng Ren <xiaofeng.ren@intel.com>, 07/2012
%
%  Edited by Qi Shan, 04/2013

function [scg_thin, scg_global, pb_orient,ret_scg_thin,ret_scg_smooth] = scg_pb_orient_Lab(img,model)
%
%function [scg_thin, scg_global] = scg_pb_orient(img,model)
%
%  scg_thin returns (thinned) local sparse code gradients
%  scg_global uses globalization and requires the spectralPb function from gPb
%

if nargout>1,
  %compute_global=1;    % edited 10/2013
  compute_global=1;
else
  compute_global=0;
end

img=im2double(img);
[h,w,ncolor]=size(img);

is_gray=(ncolor==1);
if isfield(model,'is_gray'), assert( is_gray==model.is_gray ); end

nori=model.nori;
pb_orient=zeros(h,w,nori);

if isfield(model,'svms'),
   svms=model.svms;
elseif isfield(model,'color') && isfield(model.color,'svms'),
   svms=model.color.svms;
else
   error('svms not found in model');
end

if is_gray,
  error('not supported yet');
  %img_gray=img;
  %omp_codes_gray=compute_omp_codes(img_gray,model.dic_gray);
  %for ori=1:nori,
  %  pb_orient(:,:,ori)=scg_gradient_gray(  omp_codes_gray, dic_first_gray.patchsize, ori, scales_L, models{ori}, beta(1), nori );
  %end
else
  img_gray=rgb2gray(img);
  omp_codes_gray=compute_omp_codes(img_gray,model.dic_gray);
  img_ab=RGB2Lab(img);
  img_ab=img_ab(:,:,2:3);
  omp_codes_ab=compute_omp_codes(img_ab,model.dic_ab);
  for ori=1:nori,
    pb_orient(:,:,ori)=scg_gradient_Lab(  omp_codes_gray, omp_codes_ab, model.dic_gray.patchsize, model.dic_ab.patchsize, model.scales_gray, model.scales_ab, ori, svms{ori}, model.beta(1), model.discratio, model.nori );
  end
end
pb_orient=double(pb_orient);

pb_orient=max( min( (pb_orient+1.0)/3, 1-0.05 ), 0 );
[scg_thin,scg_smooth]=smooth_gPb( pb_orient,4,3 );

if compute_global,
    sPb = spectralPb(scg_thin, [size(pb_orient,1) size(pb_orient,2)]);
    scg_smooth=0.5*scg_smooth+0.5*sqrt(abs(sPb)).*sign(sPb)/10;
    scg_smooth=max(0,min(1,scg_smooth))*(1-0.05);
    scg_global= max(scg_smooth,[],3) .* (scg_thin>0.0);
end


kk=2.5;  % kk=2.5 bring the distribution of output similar to that of gPb, no effect on benchmarking

scg_thin = scg_thin .* bwmorph(scg_thin, 'skel', inf);
scg_thin=scg_thin.^kk;
%scg_smooth=scg_smooth.^kk;
if compute_global,
  scg_global = scg_global .* bwmorph(scg_global, 'skel', inf);
  scg_global=scg_global.^kk;
end

ret_scg_thin = scg_thin;
ret_scg_smooth = scg_smooth;

end




