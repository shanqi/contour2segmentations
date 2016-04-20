
function normal=compute_surface_normal(depth)
%
% function normal=compute_surface_normal(depth)
%
%

depth=double(depth);
pcloud=DepthtoCloud(depth);
normal=pcnormal(pcloud,0.05,8);

