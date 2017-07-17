require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
require 'cudnn';
require 'cunn';
require 'cutorch';

local Gaussian,parent = torch.class('nn.Gaussian','nn.Module')

function Gaussian:__init(mean, stdv)
  parent.__init(self)
  self.mean = mean or 0
  self.stdv = stdv or 1
end

function Gaussian:updateOutput(inp)
	self.output:resizeAs(inp)
	self.output:normal(self.mean,self.stdv)
	return self.output
end

function Gaussian:updateGradInput(inp, gradOut)
	self.gradInput:resizeAs(inp)
	self.gradInput:zero()
	return self.gradInput
end

--encoder
encoder = nn.Sequential();
encoder:add(nn.View(-1,1,28,28))
encoder:add(nn.SpatialConvolution(1,32,3,3,1,1,1,1))
encoder:add(nn.ReLU())
encoder:add(nn.SpatialMaxPooling(2,2,2,2,1,1))
encoder:add(nn.SpatialConvolution(32,32,3,3,1,1,1,1))
encoder:add(nn.ReLU())
encoder:add(nn.SpatialMaxPooling(2,2,2,2))
encoder:add(nn.View(-1,1568))
encoder:add(nn.Linear(1568,784))
encoder:add(nn.BatchNormalization(784))
encoder:add(nn.ReLU())
encoder:add(nn.Linear(784,128))
encoder:add(nn.BatchNormalization(128))
encoder:add(nn.ReLU())
encoder:add(nn.Linear(128,64))
encoder:add(nn.BatchNormalization(64))
encoder:add(nn.ReLU())

local zLayer = nn.ConcatTable()
zLayer:add(nn.Linear(64,10))
zLayer:add(nn.Linear(64,10))
encoder:add(zLayer)

local noiseModule = nn.Sequential()
local noiseModuleInternal = nn.ConcatTable()
local stdModule = nn.Sequential()
stdModule:add(nn.MulConstant(0.5))
stdModule:add(nn.Exp())
noiseModuleInternal:add(stdModule)
noiseModuleInternal:add(nn.Gaussian(0,1))
noiseModule:add(noiseModuleInternal)
noiseModule:add(nn.CMulTable())

local sampler = nn.Sequential()
local samplerInternal = nn.ParallelTable()
samplerInternal:add(nn.Identity())
samplerInternal:add(noiseModule)
sampler:add(samplerInternal)
sampler:add(nn.CAddTable())

--decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(10, 64))
decoder:add(nn.BatchNormalization(64))
decoder:add(nn.ReLU())
decoder:add(nn.Linear(64, 128))
decoder:add(nn.BatchNormalization(128))
decoder:add(nn.ReLU())
decoder:add(nn.Linear(128, 784))
decoder:add(nn.ReLU())
decoder:add(nn.Linear(784, 1568))
decoder:add(nn.ReLU())
decoder:add(nn.View(32,7, 7))
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
autoencoder:add(encoder)
autoencoder:add(sampler)
autoencoder:add(decoder)
autoencoder=autoencoder:cuda()
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

	-- Optimise Gaussian KL divergence between inference model and prior: DKL[q(z|x)||N(0, σI)] = log(σ2/σ1) + ((σ1^2 - σ2^2) + (μ1 - μ2)^2) / 2σ2^2
	local nElements = xHat:nElement()
	local mean, logVar_sq = table.unpack(encoder.output)
	local var_sq = torch.exp(logVar_sq)
	local KLLoss = 0.5 * torch.sum(torch.pow(mean, 2) + var_sq - logVar_sq - 1)
	KLLoss = KLLoss/nElements
	loss = loss + KLLoss
	local gradKLLoss = {mean / nElements, 0.5*(var_sq - 1) / nElements} 
	encoder:backward(x,gradKLLoss)
	return loss, gradTheta
end

--Train
batchSize = 4000
iterations = 25
print('Training Starting')
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
	gnuplot.pngfigure('VarAE/Training_2.png')
	gnuplot.plot(table.unpack(plots))
	gnuplot.ylabel('Loss')
	gnuplot.xlabel('Batch #')
	gnuplot.plotflush()

	--permute training data
	trainData = trainData:index(1,torch.randperm(trainData:size(1)):long())
end

print('Testing')
autoencoder:evaluate()
x = testData:narrow(1,1,10)
local xHat= autoencoder:forward(x)

image.save('VarAE/Reconstructions_2.png', torch.cat(image.toDisplayTensor(x,2,10),image.toDisplayTensor(xHat,2,10),1))