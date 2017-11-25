clear;
load('AdvAE/mnist_dict_with_mappingMat.mat');
dict_modif = max(dict,0);
new_dict = dict_modif*mappingMat;
colors = reshape(randperm(30),10,3);  colors = colors/max(colors(:));
x = linspace(1,10,10);
y = linspace(1,10,10);
figure(2);scatter(x,y,100,colors,'filled');
color_points(:,:) = colors(label(:),:);
figure(1);scatter(new_dict(:,1),new_dict(:,2),30,color_points,'filled')