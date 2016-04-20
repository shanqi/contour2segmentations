%
%  Xiaofeng Ren <xiaofeng.ren@intel.com>, 07/2012
%

function [pb_thin,pb_smooth]=smooth_gPb( pb_orient, scale, margin, ratio )
%
% function [pb_thin,pb_smooth]=smooth_gPb( pb_orient, scale )
%
%    use gpb functions to smooth pb_orient
%    code extracted from multiscalePb.m in gPb by Michael Maire and Pablo Arbelaez, 2010
%

if nargin<2, scale=5; end;
if nargin<3, margin=0; end;
if nargin<4, ratio=3; end;

[h,w,nori]=size(pb_orient);
assert( nori==8 );

pb_smooth=zeros(h,w,nori);

gtheta = [1.5708    1.1781    0.7854    0.3927   0    2.7489    2.3562    1.9635];
filters = make_filters( scale, gtheta, ratio);
for o = 1 : nori,
    pb_smooth(:,:,o) = fitparab(pb_orient(:,:,o),scale,scale/ratio,gtheta(o),filters{1,o});
end

pb_thin = nonmax_channels(pb_smooth);
pb_thin = max( min( pb_thin, 1 ), 0 );

if margin>0,
  % suppress around image boundaries
  d=zeros(h,w);
  d(1:margin,:)=max( d(1:margin,:), repmat( (margin:-1:1)',[1 w] ) );
  d(:,1:margin)=max( d(:,1:margin), repmat( (margin:-1:1),[h 1] ) );
  d(end-margin+1:end,:)=max( d(end-margin+1:end,:), repmat( (1:margin)',[1 w] ) );
  d(:,end-margin+1:end)=max( d(:,end-margin+1:end), repmat( (1:margin),[h 1] ) );
  d=d/margin*10;
  pb_thin=pb_thin.*exp(-d);
  pb_smooth=pb_smooth.*repmat(exp(-d),[1 1 nori]);
end


function filters = make_filters(radii, gtheta, ratio)

d = 2;

filters = cell(numel(radii), numel(gtheta));
for r = 1:numel(radii),
    for t = 1:numel(gtheta),

        ra = radii(r);
        rb = ra / ratio;
        theta = gtheta(t);

        ra = max(1.5, ra);
        rb = max(1.5, rb);
        ira2 = 1 / ra^2;
        irb2 = 1 / rb^2;
        wr = floor(max(ra, rb));
        wd = 2*wr+1;
        sint = sin(theta);
        cost = cos(theta);

        % 1. compute linear filters for coefficients
        % (a) compute inverse of least-squares problem matrix
        filt = zeros(wd,wd,d+1);
        xx = zeros(2*d+1,1);
        for u = -wr:wr,
            for v = -wr:wr,
                ai = -u*sint + v*cost; % distance along major axis
                bi = u*cost + v*sint; % distance along minor axis
                if ai*ai*ira2 + bi*bi*irb2 > 1, continue; end % outside support
                xx = xx + cumprod([1;ai+zeros(2*d,1)]);
            end
        end
        A = zeros(d+1,d+1);
        for i = 1:d+1,
            A(:,i) = xx(i:i+d);
        end

        % (b) solve least-squares problem for delta function at each pixel
        for u = -wr:wr,
            for v = -wr:wr,
                ai = -u*sint + v*cost; % distance along major axis
                bi = u*cost + v*sint; % distance along minor axis
                if (ai*ai*ira2 + bi*bi*irb2) > 1, continue; end % outside support
                yy = cumprod([1;ai+zeros(d,1)]);
                filt(v+wr+1,u+wr+1,:) = A\yy;
            end
        end

        filters{r,t}=filt;
    end
end

