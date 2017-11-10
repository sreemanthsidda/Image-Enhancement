# Masterthesis
Image Enhancement Technique for Forensic Evidence
%%%image enhancement using Sparse Representation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Final_maincode()
clc
clear all 
close all
shoe_inp=imread(uigetfile('*.jpg')); %loading the shoeprint image
warning off
shoe_inp=uint16(shoe_inp);
I=imresize(shoe_inp,[512 512]); %resizing the input image
if ndims(I)==3
    I=rgb2gray(I);  
end
%im=im2double(im);
[M,N]=size(I); %size of the image
imshow(mat2gray(I)),title('Original Image');
% im=imadjust(im,stretchlim(im),[]);
% im1=histeq(uint8(I));
t=imsmqt(uint8(I),8);
colormap gray
figure,imshow(t);
title('SMQT Image');
im=wiener2(t);
%% Eqn 2

[cA,cH,cV,cD]  = dwt2(im,'haar');
figure,imshow([cA cH; cV cD],[]);
title('DWT 1st level');

[LL,LH,HL,HH] = dwt2(im,'db4');
% figure,
% subplot(221),imshow(LL),title('Low-low subband')
%  subplot(222),imshow(LH),title('Low-high subband')
%  subplot(223),imshow(HL),title('High-low subband')
%  subplot(224),imshow(HH),title('High-high subband')
%% Eqn 3
%ll_idx=find(LL>0);
%lh_idx=find(LH>0);
%hl_idx=find(HL>0);
%hh_idx=find(HH>0);
mean_ll=0;
mean_lh=0;
mean_hl=0;
mean_hh=0;
for i=1:size(LL,1)
    for j=1:size(LL,2)
        if (LL(i,j)>=0)
            mean_ll=mean_ll+LL(i,j);
        end
    end
end
mean_ll=mean_ll*4/numel(im);

for i=1:size(LH,1)
    for j=1:size(LH,2)
        if (LH(i,j)>=0)
            mean_lh=mean_lh+LH(i,j);
        end
    end
end
mean_lh=mean_lh*4/numel(im);

for i=1:size(HL,1)
    for j=1:size(HL,2)
        if (HL(i,j)>=0)
            mean_hl=mean_hl+HL(i,j);
        end
    end
end
mean_hl=mean_hl*4/numel(im);

for i=1:size(HH,1)
    for j=1:size(HH,2)
        if (HH(i,j)>=0)
            mean_hh=mean_hh+HH(i,j);
        end
    end
end
mean_hh=mean_hh*4/numel(im);



%% Eqn 4
mean_max=max([mean_ll,mean_lh,mean_hl,mean_hh]);
%% Eqn 5
[u_ll,s_ll,v_ll]=svd(LL);
[u_lh,s_lh,v_lh]=svd(LH); 
[u_hl,s_hl,v_hl]=svd(HL);
[u_hh,s_hh,v_hh]=svd(HH);
%% Gaussian 2 D matrix
G=fspecial('gaussian',size(im));
G=imfilter(im,G);
[G_LL,G_LH,G_HL,G_HH]=dwt2(G,'db4');
%% Eqn 6
[ug_ll,sg_ll,vg_ll]=svd(G_LL);
[ug_lh,sg_lh,vg_lh]=svd(G_LH);
[ug_hl,sg_hl,vg_hl]=svd(G_HL);
[ug_hh,sg_hh,vg_hh]=svd(G_HH);
%% Eqn 7 8 9 10
weight_ll=(mean_max/mean_ll)*(max(max(sg_ll))/max(max(LL)));
weight_hl=((mean_max/mean_hl)*(max(max(sg_hl))/max(max(HL))))^(1/6);
weight_lh=((mean_max/mean_lh)*(max(max(sg_lh))/max(max(LH))))^(1/6);
weight_hh=((mean_max/mean_hh)*(max(max(sg_hh))/max(max(HH))))^(1/8);
%% Eqn 11
new_LL=1*(u_ll*(weight_ll.*s_ll)*v_ll');
new_LH=1*(u_lh*(weight_lh.*s_lh)*v_lh');
new_HL=1*(u_hl*(weight_hl.*s_hl)*v_hl');
new_HH=1*(u_hh*(weight_hh.*s_hh)*v_hh');
%% Eqn 12
newim=idwt2(new_LL,new_LH,new_HL,new_HH,'db4');
 newim=im2double(newim);
figure,imshow(newim,[]),title('SVD Method')
n_svd=uint8(newim);
figure,
subplot(221),imshow(I),title('Original Image')
he_im=histeq(uint8(I));
%% Calculation of MSE and PSNR
I = uint8(I);
MSE1=sum(sum((double(t)-double(I)).^2))/(M*N); 
fprintf('\n The MSE value of Pre-processed image is %0.4f', MSE1);
psnr1=10*log10(255^2./MSE1);
MSE2=(sum(sum(double((I-n_svd)).*double((I-n_svd))))/(numel(n_svd)));
fprintf('\n The MSE value of SVD processed image is %0.8f', MSE2);
psnr2=10*log10(255^2./MSE2);
psnr1=abs(psnr1);
psnr2=abs(psnr2);
fprintf('\n The Peak-SNR value of Pre-processed image is %0.4f', psnr1);
fprintf('\n The Peak-SNR value SVD processed image is %0.4f', psnr2);
subplot(222),
imshow(he_im),title('Hist Eq Image')
im=imsmqt(t,8);
colormap gray
subplot(223)
imshow(im),title('SMQT Image')
% newim(im==255)=9000;
N=newim;
subplot(224),imshow(N,[]),title('SVD Method')

%% Calculation of SSIM
 K = [0.01 0.03];
   window = ones(8);
   L = 100;
   N_sim=uint8(N);
   I_sim=uint8(I);
  [mssim, ssim_map] = ssim(N_sim,I_sim, K, window, L);
  fprintf('\n The SSIM value is %f \n',mssim);

%%%%%% Sparse Representation  %%%%%%%%%
input_image = im2double(mat2gray(N));
im_withnoise=N;
[~,~] = size(im_withnoise);
% obtain the Gaussian Filter with the size 5*5

% imshow(im_withoutnoise_gaussian,[])

    mean_matrix_8x8 = zeros( 8,8 );

for m = 0:63
      for n = 0:63
         blk_matrix_8x8 = input_image(m*8+[1:8],n*8+[1:8]);        
         fl_im=filter2enhance(blk_matrix_8x8,[1]);
         img_gaussian_enhanced(m*8+[1:8],n*8+[1:8])= fl_im;              
      end
end
img_gaussian_enhanced=imadjust(img_gaussian_enhanced);
imn_enh=img_gaussian_enhanced;
  figure,imshow(imn_enh,[])
  title('Enhanced shoeprint image(SVD+SPARSE)')
%   figure,newsharp=imsharpen(imn_enh); imshow(newsharp,[])
%   title('Enh image')
%%%%PSNR calculation
% ky=input_image; 
I= im2double(mat2gray(I));
kt=img_gaussian_enhanced;
mse_enh=(sum(sum(double((kt-I)).*double((kt-I))))/numel(I)); %normalised MSE
 m=size(kt);  
 fprintf('\n MSE value of enhanced image is')
 disp(mse_enh)
 psnr=10*log10((255^2)/(mse_enh));
 fprintf('PSNR value of enhanced image is')
 disp(psnr) % displaying the enhanced image PSNR
figure;
subplot(311);
imhist(I);
title('Histogram of Original Image');
subplot(312);
im_svd=im2double(mat2gray(newim));
imhist(im_svd);
title('Histogram of SVD processed Image');
subplot(313);imhist(imn_enh);
title('Histogram of Final Enhanced Image');
% Fi=imsharpen(imn_enh);
% figure;imshow(Fi);
% im_spar=uint8(imn_enh);
ky=input_image;

%%% Gaussian Filter %%%
 function b = filter2enhance(varargin)

[a,h,boundary,flags] = parse_inputs(varargin{:});
  
rank_a = ndims(a);
rank_h = ndims(h);
  
% Pad dimensions with ones if filter and image rank are different

size_a = [size(a) ones(1,rank_h-rank_a)];
size_h = [size(h) ones(1,rank_a-rank_h)];
if bitand(flags,8)
  %Full output
  im_size = size_a+size_h-1;
  pad = size_h - 1;
else
  %Same output
  im_size = size_a;

  %Calculate the number of pad pixels
  filter_center = floor((size_h + 1)/2);
  pad = size_h - filter_center;
end

%Empty Inputs
% 'Same' output then size(b) = size(a)
% 'Full' output then size(b) = size(h)+size(a)-1 
if isempty(a)
  if bitand(flags,4) %Same
    b = a;
  else %Full
    if all(im_size>0)
      b = a;
      b = b(:);
      b(prod(im_size)) = 0;
      b = reshape(b,im_size);
    elseif all(im_size>=0)
      b = feval(class(a),zeros(im_size));
    else
      eid = sprintf('Images:%s:negativeDimensionBadSizeB',mfilename);
      msg = ['Error in size of B.  At least one dimension is negative. ',...
             '\n''Full'' output size calculation is: size(B) = size(A) ',...
             '+ size(H) - 1.'];
      error(eid,msg);
    end
  end
  return;
end

if  isempty(h)
  if bitand(flags,4) %Same
    b = a;
    b(:) = 0;
  else %Full
    if all(im_size>0)
      b = a;
      if all(im_size<size_a)  %Output is smaller than input
        b(:) = [];
      else %Grow the array, is this a no-op?
        b(:) = 0;
        b = b(:);
      end
      b(prod(im_size)) = 0;
      b = reshape(b,im_size);
    elseif all(im_size>=0)
      b = feval(class(a),zeros(im_size));
    else
      eid = sprintf('Images:%s:negativeDimensionBadSizeB',mfilename);
      msg = ['Error in size of B.  At least one dimension is negative. ',...
             '\n''Full'' output size calculation is: size(B) = size(A) +',...
             ' size(H) - 1.'];
      error(eid,msg);
    end
  end
  return;
end

im_size = im_size;

%Starting point in padded image, zero based.
start = pad;

%Pad image
a = padarray(a,pad,boundary,'both');

separable = false;
numel_h = numel(h);
if isa(a,'double')
  sep_threshold = (numel_h >= 49);
else
  sep_threshold = (numel_h >= 289);
end

if sep_threshold && (rank_a == 2) && (rank_h == 2) && ...
      all(size(h) ~= 1) && ~any(isnan(h(:))) && ~any(isinf(h(:)))
  [u,s,v] = svd(h);
  s = diag(s);
  tol = length(h) * max(s) * eps;
  rank = sum(s > tol);
  if (rank == 1)
    separable = true;
  end
end

if separable
  % extract the components of the separable filter
  hcol = u(:,1) * sqrt(s(1));
  hrow = v(:,1)' * sqrt(s(1));
  
  % Create connectivity matrix.  Only use nonzero values of the filter.
  conn_logical_row = hrow~=0;
  conn_row = double(conn_logical_row); %input to the mex file must be double
  nonzero_h_row = hrow(conn_logical_row);

  conn_logical_col = hcol~=0;
  conn_col = double(conn_logical_col); %input to the mex file must be double
  nonzero_h_col = hcol(conn_logical_col);

  % intermediate results should be stored in doubles in order to
  % maintain sufficient precision
  class_of_a = class(a);
  change_class = false;
  if ~strcmp(class_of_a,'double')
    change_class = true;
    a = double(a);
  end

  % apply the first component of the separable filter (hrow)
  checkMexFileInputs(a,im_size,real(hrow),real(nonzero_h_row),conn_row,...
                     start,flags);  
  b_row_applied = filter_mex(a,im_size,real(hrow),real(nonzero_h_row),...
                               conn_row,start,flags);

  if ~isreal(hrow) %imaginary
    b_row_applied_cmplx = imfilter_mex(a,im_size,imag(hrow),...
                                       imag(nonzero_h_row),conn_row,...
                                       start,flags);    
    if isreal(a)
      % b_row_applied and b_row_applied_cmplx will always be real;
      % result will always be complex
      b_row_applied = complex(b_row_applied,b_row_applied_cmplx);
    else 
      % b_row_applied and/or b_row_applied_cmplx may be complex;
      % result will always be complex
      b_row_applied = complex(imsubtract(real(b_row_applied),...
                                         imag(b_row_applied_cmplx)),...
                              imadd(imag(b_row_applied),...
                                    real(b_row_applied_cmplx)));
    end
  end
  
 
  b_row_applied = padarray(b_row_applied,pad,boundary,'both');
  
  checkMexFileInputs(b_row_applied,im_size,real(hcol),real(nonzero_h_col),...
                     conn_col,start,flags);
  b1 = imfilter_mex(b_row_applied,im_size,real(hcol),real(nonzero_h_col),...
                    conn_col,start,flags);

  if ~isreal(hcol)
    b2 = imfilter_mex(b_row_applied,im_size,imag(hcol),imag(nonzero_h_col),...
                      conn_col,start,flags);
    if change_class
      b2 = feval(class_of_a,b2);
    end
  end
  
  % change the class back if necessary  
  if change_class
    b1 = feval(class_of_a,b1);
  end
  
  %If input is not complex, the output should not be complex. COMPLEX always
  %creates an imaginary part even if the imaginary part is zeros.
  if isreal(hcol)
    % b will always be real
    b = b1;
  elseif isreal(b_row_applied)
    % b1 and b2 will always be real. b will always be complex
    b = complex(b1,b2);
  else
    % b1 and/or b2 may be complex.  b will always be complex
    b = complex(imsubtract(real(b1),imag(b2)),imadd(imag(b1),real(b2)));
  end

else % non-separable filter case
  
  % Create connectivity matrix.  Only use nonzero values of the filter.
  conn_logical = h~=0;
  conn = double( conn_logical );  %input to the mex file must be double
  
  nonzero_h = h(conn_logical);
  
  
  checkMexFileInputs(a,im_size,real(h),real(nonzero_h),conn,start,flags);
  b1 = filter_mex(a,im_size,real(h),real(nonzero_h),conn,start,flags);
  
  if ~isreal(h)
    checkMexFileInputs(a,im_size,imag(h),imag(nonzero_h),conn,start,flags);
    b2 = imfilter_mex(a,im_size,imag(h),imag(nonzero_h),conn,start,flags);
  end
  
  %If input is not complex, the output should not be complex. COMPLEX always
  %creates an imaginary part even if the imaginary part is zeros.
  if isreal(h)
    % b will always be real
    b = b1;
  elseif isreal(a)
    % b1 and b2 will always be real. b will always be complex
    b = complex(b1,b2);
  else
    % b1 and/or b2 may be complex.  b will always be complex
    b = complex(imsubtract(real(b1),imag(b2)),imadd(imag(b1),real(b2)));
    
  end
end

%======================================================================

function [a,h,boundary,flags ] = parse_inputs(a,h,varargin)

iptchecknargin(2,5,nargin,mfilename);

iptcheckinput(a,{'numeric' 'logical'},{'nonsparse'},mfilename,'A',1);
iptcheckinput(h,{'double'},{'nonsparse'},mfilename,'H',2);

%Assign defaults
flags = 0;
boundary = 0;  %Scalar value of zero
output = 'same';
do_fcn = 'corr';

allStrings = {'replicate', 'symmetric', 'circular', 'conv', 'corr', ...
              'full','same'};

for k = 1:length(varargin)
  if ischar(varargin{k})
    string = iptcheckstrs(varargin{k}, allStrings,...
                          mfilename, 'OPTION',k+2);
    switch string
     case {'replicate', 'symmetric', 'circular'}
      boundary = string;
     case {'full','same'}
      output = string;
     case {'conv','corr'}
      do_fcn = string;
    end
  else
    iptcheckinput(varargin{k},{'numeric'},{'nonsparse'},mfilename,'OPTION',k+2);
    boundary = varargin{k};
  end %else
end

if strcmp(output,'full')
  flags = bitor(flags,8);
elseif strcmp(output,'same');
  flags = bitor(flags,4);
end

if strcmp(do_fcn,'conv')
  flags = bitor(flags,2);
elseif strcmp(do_fcn,'corr')
  flags = bitor(flags,0);
end


%--------------------------------------------------------------
function checkMexFileInputs(varargin)
% a
a = varargin{1};
iptcheckinput(a,{'numeric' 'logical'},{'nonsparse'},mfilename,'A',1);

% im_size
im_size = varargin{2};
if ~strcmp(class(im_size),'double') || issparse(im_size)
  displayInternalError('im_size'); 
end

% h
h = varargin{3};
if ~isa(h,'double') || ~isreal(h) || issparse(h)
  displayInternalError('h');
end

% nonzero_h
nonzero_h = varargin{4};
if ~isa(nonzero_h,'double') || ~isreal(nonzero_h) || ...
      issparse(nonzero_h)
  displayInternalError('nonzero_h');
end

% start
start = varargin{6};
if ~strcmp(class(start),'double') || issparse(start)
  displayInternalError('start');
end

% flags
flags = varargin{7};
if ~isa(flags,'double') ||  any(size(flags) ~= 1)
  displayInternalError('flags');
end 

%--------------------------------------------------------------
function displayInternalError(string)

eid = sprintf('Images:%s:internalError',mfilename);
msg = sprintf('Internal error: %s is not valid.',upper(string));
error(eid,'%s',msg);


