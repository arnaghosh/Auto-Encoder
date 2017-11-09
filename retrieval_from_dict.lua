require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
--require 'cutorch';
--require 'cunn';
--require 'cudnn';
require './BinarizedNeurons'

zSize = 14
--encoder
encoder = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/AdvAE_data/encoder1.t7');

binariser = nn.Sequential();
binariser:add(BinarizedNeurons())

autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(binariser)

autoencoder = autoencoder--:cuda()
print(autoencoder)

--load MNIST data
trainData = mnist.traindataset().data:double():div(255):reshape(60000,1,28,28)--:cuda()
trainlabels = (mnist.traindataset().label+1)--:cuda()
N = mnist.traindataset().size

testData = mnist.testdataset().data:double():div(255):reshape(10000,1,28,28)--:cuda()
testlabels = (mnist.testdataset().label+1)--:cuda()
teSize = mnist.testdataset().size
print(N,teSize)

local dict = torch.load('AdvAE/mnist_dict.t7');

local x,y

batchSize = 3000
iterations = 50
num_retrieval = 5;
retrieval_vec = torch.Tensor(teSize,num_retrieval+1)

--Train
print('Dictionary retrieving')

dict_val = torch.Tensor(dict:size(1))
for n=1,teSize do
	collectgarbage()
	x = testData[n];
	--print(x:size())
	x1 = autoencoder:forward(x)
	for i=1,dict:size(1) do
		dict_val[i] = x1:dot(dict[i])
	end
	_,index = torch.sort(dict_val,true)
	retrieval_vec[n][1] = testlabels[n];
	retrieval_vec[{{n},{2,num_retrieval+1}}] = trainlabels:index(1,index[{{1,num_retrieval}}]:long());
end

torch.save('AdvAE/mnist_retrieval.t7',retrieval_vec)