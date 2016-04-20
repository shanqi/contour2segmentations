
function convert_dataset

% download nyu depth v2 dataset
if ~exist('nyu_depth_v2_labeled.mat'),
  system('wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat');
end

load list_train.txt
load list_test.txt

% generate half-size images

load nyu_depth_v2_labeled.mat images
nframe=size(images,4);
mask_border=find_border_region(images);

disp('generating color images...');
generate_image_half(images,list_train,'train');
generate_image_half(images,list_test,'test');
clear images

load nyu_depth_v2_labeled.mat depths
disp('generating depth images...');
generate_depth_half(depths,list_train,'train');
generate_depth_half(depths,list_test,'test');
clear depths

% generate cleaned-up groundtruth in BSDS format
disp('generating groundtruth... could take a while');
load nyu_depth_v2_labeled.mat labels instances
generate_groundtruth_medfilt_half(labels,instances,mask_border,list_train,'train');
generate_groundtruth_medfilt_half(labels,instances,mask_border,list_test,'test');

% DONE.
disp('DONE.');


function mask_border=find_border_region(images)

[im_h,im_w,~,nframe]=size(images);

mask_border=ones(im_h,im_w);
for ii=1:nframe,
  im=rgb2gray(images(:,:,:,ii));
  mask_border=(mask_border & (im>=253));
end


function generate_image_half(images,list,dirname)

outdir=['./data/images/' dirname];
if exist(outdir), return; end;

system(['mkdir -p ' outdir]);
for ii=list',
  id=num2str(ii,'%08d');
  img=images(:,:,:,ii);
  img=imresize(img,0.5);
  imwrite(img,[outdir '/' id '.jpg'],'Quality',98);
end


function generate_depth_half(depths,list,dirname)

outdir=['./data/depth/' dirname];
if exist(outdir), return; end;

system(['mkdir -p ' outdir]);
for ii=list',
  id=num2str(ii,'%08d');
  depth=depths(:,:,ii);
  depth=depth(2:2:end,2:2:end);
  imwrite(uint16(depth*1000), [outdir '/' id '.png']);
end


function generate_groundtruth_medfilt_half(labels,instances,mask_border,list,dirname)

outdir=['./data/groundTruth/' dirname];
if exist(outdir), return; end;

maxlabel=1000; assert( max(labels(:))<=maxlabel );

system(['mkdir -p ' outdir]);
for ii=list',
  id=num2str(ii,'%08d');
  % find out the region that needs to be filled, use dilate/erode
  margin=5;
  foreground=(labels(:,:,ii)>0);
  foreground=imerode(imdilate(foreground,strel('disk',margin)),strel('disk',margin));
  cc=bwconncomp(~foreground);
  A=regionprops(cc,'Area');
  min_background_area=500;
  indA=find([A.Area]<min_background_area);
  for k=indA, foreground(cc.PixelIdxList{k})=1; end;
  % now do repeated median filter to fill in foreground
  seg=double(instances(:,:,ii))*maxlabel+double(labels(:,:,ii));
  seg0=seg;
  seg(seg==0)=nan;
  for iter=1:100,
    seg2=mediannan_int(seg,5);
    ind=find(isnan(seg) & ~isnan(seg2) & foreground);
    if isempty(ind), break; end;
    seg(ind)=seg2(ind);
  end
  seg(isnan(seg))=0;

  %groundTruth{1}.Segmentation=uint16(seg .* (1-mask_border));    % objects start with 1; 0 is background
  %groundTruth{1}.Boundaries=logical(seg2bdry(seg,'imageSize')) & ~imdilate(mask_border,strel('disk',5));
  %save([outdir '/' id '.mat'],'groundTruth');

  groundTruth{1}.Segmentation=uint16(seg(2:2:end,2:2:end).*(1-mask_border(2:2:end,2:2:end)));   % objects start with 1; 0 is background
  groundTruth{1}.Boundaries=logical(seg2bdry(seg(2:2:end,2:2:end),'imageSize')) & ~imdilate(mask_border(2:2:end,2:2:end),strel('disk',2));
  save([outdir '/' id '.mat'],'groundTruth');
end




