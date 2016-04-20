%
%  Xiaofeng Ren <xiaofeng.ren@intel.com>, 12/2012
%


dataset='test';

% download the nyu dataset (v2, bsds fomat) separately
imgdir=['./nyu_v2/data/images/' dataset '/'];

files=dir([imgdir '/*jpg']);

addpath(genpath('./omp'));
addpath ./gpb

load scg_model_RGBD;

nfile=length(files);

run_local=0;
run_global=~run_local;

if run_local
    outdir=['scg_color_local_nyu2/' dataset '/'];
else
    outdir=['scg_color_global_nyu2/' dataset '/'];
end
system(['mkdir -p ' outdir]);


ncpu=1; icpu=1;
for ifile=icpu:ncpu:nfile,
%for ifile=nfile-icpu+1:-ncpu:1,
  id=files(ifile).name(1:end-4);
  if ~exist([outdir '/' id '.mat']),
    tic;
    img=imread([imgdir '/' id '.jpg']);
    if run_local,
      [scg_thin]=scg_pb_orient_Lab(img,scg_model_RGBD);
      gPb_thin=single(scg_thin);
    else
      [scg_thin,scg_global]=scg_pb_orient_Lab(img,scg_model_RGBD);
      gPb_thin=single(scg_global);
    end
    gPb_thin=gPb_thin.^0.6;  % no real impact; to do precision-recall with fewer points
    save([outdir '/' id '.mat'],'gPb_thin');
    tt=toc;
  end
  disp(num2str([ifile nfile tt]));
end


