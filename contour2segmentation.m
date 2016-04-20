function contour2segmentation(pball_filename)

pbfile = fopen(pball_filename);
pbsize = fread(pbfile, [1,4], 'int32');
pbdata = fread(pbfile,[pbsize(4),pbsize(3)*pbsize(2)],'float');
pbdata = reshape(pbdata,[pbsize(4),pbsize(3),pbsize(2)]);

[ucm] = contours2ucm(pbdata, 'doubleSize');

%%

k = 0.9;
% bdry = (ucm >= k);
% get superpixels at scale k without boundaries:
labels2 = bwlabel(ucm <= k);
labels = labels2(2:2:end, 2:2:end);
%figure;imshow(labels,[]); colormap(jet);

%labels(labels>255) = 255;

strk = strfind(pball_filename,'.');
outfilename = [pball_filename(1:strk(1)), 'seg.png'];
imwrite(ind2rgb(labels,jet), outfilename);

end
