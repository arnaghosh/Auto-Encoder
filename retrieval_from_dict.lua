require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
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

autoencoder = autoencoder:cuda()
print(autoencoder)

--load MNIST data
trainData = mnist.traindataset().data:double():div(255):reshape(60000,1,28,28)
trainlabels = (mnist.traindataset().label+1)
N = mnist.traindataset().size

testData = mnist.testdataset().data:double():div(255):reshape(10000,1,28,28):cuda()
testlabels = (mnist.testdataset().label+1):cuda()
teSize = mnist.testdataset().size
print(N,teSize)

local dict = torch.load('AdvAE/mnist_dict.t7');

local x,y

batchSize = 3000
iterations = 50
num_retrieval = 5;
retrieval_vec = torch.Tensor(teSize,3)

--Train
print('Dictionary retrieving')
x1 = autoencoder:forward(testData)
testData = nil;
autoencoder = nil;
dict = dict:cuda()
dict_val = torch.Tensor(dict:size(1))
dict_val = torch.mm(x1,dict:transpose(1,2))
for n=1,teSize do
	collectgarbage()
	if n%1000==0 then print("yeah") end
	--x = testData[n];
	--print(x:size())
	--x1 = autoencoder:forward(x)
	retrieval_vec[n][1] = testlabels[n];
	retrieval_vec[n][2] = 0;
	retrieval_vec[n][3] = 0;
	for i=1,dict:size(1) do
		--dict_val[i] = x1:dot(dict[i]:cuda())
		if (dict_val[n][i]>=zSize-4) then  --- Hamming Distance<=2 --> dot product val>=zSize-4
			retrieval_vec[n][2] = retrieval_vec[n][2]+1;
			if(trainlabels[i]==testlabels[n]) then
				retrieval_vec[n][3] = retrieval_vec[n][3]+1;
			end
		end
	end
	--_,index = torch.sort(dict_val,true)
	--retrieval_vec[{{n},{2,num_retrieval+1}}] = trainlabels:index(1,index[{{1,num_retrieval}}]:long());
end

torch.save('AdvAE/mnist_retrieval.t7',retrieval_vec)
