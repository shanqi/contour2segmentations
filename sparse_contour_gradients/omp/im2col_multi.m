function X = im2col_multi(im, grid, type, c)

X = [];
for i = 1:c
    X_tmp = im2col(im(:,:,i), grid, type);
    X = [X; X_tmp];
end

