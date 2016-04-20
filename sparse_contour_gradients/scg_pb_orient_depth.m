%
%  Xiaofeng Ren <xiaofeng.ren@intel.com>, 01/2013
%

function [scg_thin, scg_global] = scg_pb_orient(depth,normal,model)
%
%function [scg_thin, scg_global] = scg_pb_orient(depth,normal,model)
%
%  scg_thin returns (thinned) local sparse code gradients
%  scg_global uses globalization and requires the spectralPb function from gPb
%

if nargout>1,
  compute_global=1;
else
  compute_global=0;
end

depth=double(depth);
normal=double(normal);
[h,w,~]=size(depth);

if isfield(model,'svms'),
   svms=model.svms;
elseif isfield(model,'depth') & isfield(model.depth,'svms'),
   svms=model.depth.svms;
else
   error('svms not found in model');
end

nori=model.nori;
pb_orient=zeros(h,w,nori);

  omp_codes_gray=compute_omp_codes(depth,model.dic_depth);
  omp_codes_ab=compute_omp_codes(normal,model.dic_normal);
  for ori=1:nori,
    pb_orient(:,:,ori)=scg_gradient_Lab(  omp_codes_gray, omp_codes_ab, model.dic_gray.patchsize, model.dic_ab.patchsize, model.scales_gray, model.scales_ab, ori, svms{ori}, model.beta(2), model.discratio, model.nori );
  end
pb_orient=double(pb_orient);

pb_orient=max( min( (pb_orient+1.0)/3, 1-0.05 ), 0 );
[scg_thin,scg_smooth]=smooth_gPb( pb_orient,4,3 );

if compute_global,
    [sPb] = spectralPb(scg_thin, [size(pb_orient,1) size(pb_orient,2)]);
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



