require 'nn';
require 'dpnn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
require 'cudnn';
require 'cutorch';
require 'cunn';

--encoder
encoder = nn.Sequential();
encoder:add(nn.View(-1,1,28,28))
encoder:add(nn.SpatialConvolution(1,32,3,3,1,1,1,1))
encoder:add(nn.ReLU())
encoder:add(nn.SpatialMaxPooling(2,2,2,2,1,1))
encoder:add(nn.SpatialConvolution(32,32,3,3,1,1,1,1))
encoder:add(nn.ReLU())
encoder:add(nn.SpatialMaxPooling(2,2,2,2))

--decoder
decoder = nn.Sequential()
decoder:add(nn.SpatialConvolution(32,32,3,3,1,1,1,1))
decoder:add(nn.ReLU())
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolution(32,32,3,3,1,1,1,1))
decoder:add(nn.ReLU())
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolution(32,1,3,3,1,1,1,1))
decoder:add(nn.Sigmoid())
decoder:add(nn.View(28,28))

--autoencoder
autoencoder = nn.Sequential()
noiser = nn.WhiteNoise(0,0.2)
autoencoder:add(noiser)
autoencoder:add(encoder)
autoencoder:add(decoder)

autoencoder = autoencoder:cuda()
print(autoencoder)

trainData = mnist.traindataset().data:double():div(255):cuda()
trainlabels = mnist.traindataset().label:cuda()
N = mnist.traindataset().size

testData = mnist.testdataset().data:double():div(255):cuda()
testlabels = mnist.testdataset().label:cuda()


local theta,gradTheta = autoencoder:getParameters()

local criterion = nn.BCECriterion():cuda()

local x

local feval = function(params)
	if theta~=params then
		theta:copy(params)
	end
	gradTheta:zero()
	--print(#x)
	local xHat = autoencoder:forward(x)
	local loss = criterion:forward(xHat,x)
	local gradLoss = criterion:backward(xHat,x)
	autoencoder:backward(x,gradLoss)
	--print(xHat[1])
	return loss, gradTheta
end

--Train
batchSize = 5000
iterations = 1
print('Training Starting')
autoencoder:training()
local optimParams = {learningRate = 0.1}
local _,loss 
local losses = {}
for epoch=1,iterations do
	collectgarbage()
	print('Epoch '..epoch..'/'..iterations)
	for n=1,N, batchSize do
		collectgarbage()
		x = trainData:narrow(1,n,batchSize)
		_,loss = optim.sgd(feval,theta,optimParams)
		losses[#losses + 1] = loss[1]
	end
	local plots={{'AE', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
	--[[gnuplot.pngfigure('DenoisingAE/Training_2.png')
	gnuplot.plot(table.unpack(plots))
	gnuplot.ylabel('Loss')
	gnuplot.xlabel('Batch #')
	gnuplot.plotflush()
--]]
	--permute training data
	trainData = trainData:index(1,torch.randperm(trainData:size(1)):long())
end

print('Testing')
--autoencoder:evaluate()
x = testData:narrow(1,1,50)
--print(x[1])
local xHat= autoencoder:forward(x)
x1= noiser.output
print(#encoder.output)
--print(xHat[1])
--image.save('DenoisingAE/Reconstructions_2.png', torch.cat(image.toDisplayTensor(x1,2,50),image.toDisplayTensor(xHat,2,50),1))
