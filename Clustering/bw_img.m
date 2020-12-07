img_data = im2double(imread('/Users/manasbundele/Downloads/bw.jpg'));
X_kmeans = reshape(img_data,[7500,1]);
X_spectral = reshape(img_data,[1,7500]);
%img_vec = img_data(:);
sigma = 0.10
k = 2
%C = spectral_clustering(X_spectral, k, sigma);
C = kmeans(X_kmeans,2)
output = reshape(C,[75,100])
output(output == 2) = -1
imshow(output)