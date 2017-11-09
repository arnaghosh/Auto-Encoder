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

local dict = torch.Tensor(N,zSize);

local x,y

batchSize = 3000
iterations = 50


--Train
print('Dictionary preparing')

for n=1,N,batchSize do
	collectgarbage()
	x = trainData:narrow(1,n,batchSize)--:cuda()
	--print(x:size())
	x1 = autoencoder:forward(x)
	dict[{{n,n+batchSize-1},{}}] = x1;
end

torch.save('AdvAE/mnist_dict.t7',dict)