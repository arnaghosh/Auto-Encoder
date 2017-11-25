require 'nn';
require 'image';
mnist = require 'mnist';
matio = require 'matio'
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
--require 'cudnn';

Data = torch.load('AdvAE/mnist_dict.t7')
Labels = (mnist.traindataset().label+1)
--print(dict_data:size(), trainlabels:size())
zSize = Data:size(2)

bit_imp = torch.load('AdvAE/bits_imp.t7')
bit_imp = bit_imp - bit_imp[zSize+1]

_,index = torch.sort(bit_imp:narrow(1,1,zSize),true); --true for descending order

feature_mapper = torch.zeros(zSize,2);

for b=1,zSize do
	if b%2==1 then --1st feature vector changed
		feature_mapper[index[b]][1] = 2^torch.floor((zSize-b)/2);
		feature_mapper[index[b]][2] = 0;
	else
		feature_mapper[index[b]][2] = 2^torch.floor((zSize-b)/2);
		feature_mapper[index[b]][1] = 0;
	end
end
print(index)
print(feature_mapper)

matio.save('AdvAE/mnist_dict_with_mappingMat.mat',{dict=Data,label=Labels,mappingMat=feature_mapper});