require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'cudnn';

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

zSize = 128
--encoder
encoder = nn.Sequential();
encoder:add(nn.SpatialConvolution(3,64,3,3,1,1,1,1)) --1
encoder:add(nn.ReLU()) --2
encoder:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1)) --3
encoder:add(nn.ReLU()) --4
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --5

encoder:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1)) --6
encoder:add(nn.ReLU()) --7
encoder:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1)) --8
encoder:add(nn.ReLU()) --9
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --10

encoder:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1)) --11
encoder:add(nn.ReLU()) --12
encoder:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1)) --13
encoder:add(nn.ReLU()) --14
encoder:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1)) --15
encoder:add(nn.ReLU()) --16
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --17

encoder:add(nn.SpatialConvolution(256,512,3,3,1,1,1,1)) --18
encoder:add(nn.ReLU()) --19
encoder:add(nn.SpatialConvolution(512,512,3,3,1,1,1,1)) --20
encoder:add(nn.ReLU()) --21
encoder:add(nn.SpatialConvolution(512,512,1,1,1,1)) --22
encoder:add(nn.ReLU()) --23
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --24

encoder:add(nn.View(-1,2048)) --25
encoder:add(nn.Linear(2048,512)) --26
encoder:add(nn.BatchNormalization(512)) --27
encoder:add(nn.ReLU()) --28
encoder:add(nn.Linear(512,zSize)) --29

--decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(zSize, 512)) --1
decoder:add(nn.BatchNormalization(512)) --2
decoder:add(nn.ReLU()) --3
decoder:add(nn.Linear(512, 2048)) --4
decoder:add(nn.BatchNormalization(2048)) --5
decoder:add(nn.ReLU()) --6
decoder:add(nn.View(512,2,2)) --7

decoder:add(nn.SpatialMaxUnpooling(encoder:get(24))) --8
decoder:add(nn.SpatialConvolution(512,512,1,1,1,1)) --9
decoder:add(nn.ReLU()) --10
decoder:add(nn.SpatialConvolution(512,512,3,3,1,1,1,1)) --11
decoder:add(nn.ReLU()) --12
decoder:add(nn.SpatialConvolution(512,256,3,3,1,1,1,1)) --13
decoder:add(nn.ReLU()) --14

decoder:add(nn.SpatialMaxUnpooling(encoder:get(17))) --15
decoder:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1)) --16
decoder:add(nn.ReLU()) --17
decoder:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1)) --18
decoder:add(nn.ReLU()) --19
decoder:add(nn.SpatialConvolution(256,128,3,3,1,1,1,1)) --20
decoder:add(nn.ReLU()) --21

decoder:add(nn.SpatialMaxUnpooling(encoder:get(10))) --22
decoder:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1)) --23
decoder:add(nn.ReLU()) --24
decoder:add(nn.SpatialConvolution(128,64,3,3,1,1,1,1)) --25
decoder:add(nn.ReLU()) --26

decoder:add(nn.SpatialMaxUnpooling(encoder:get(5))) --27
decoder:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1)) --28
decoder:add(nn.ReLU()) --29
decoder:add(nn.SpatialConvolution(64,3,1,1,1,1)) --30  -- Used a 1x1 convolution instead of 3x3 -- Make change across run
decoder:add(nn.Sigmoid()) --31


--autoencoder
autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(decoder)

autoencoder = autoencoder:cuda()
print(autoencoder)

--adversary network
adversary = nn.Sequential()
adversary:add(nn.Linear(zSize, 64))
adversary:add(nn.BatchNormalization(64))
adversary:add(nn.ReLU())
adversary:add(nn.Linear(64, 16))
adversary:add(nn.BatchNormalization(16))
adversary:add(nn.ReLU())
adversary:add(nn.Linear(16, 1))
adversary:add(nn.BatchNormalization(1))
adversary:add(nn.Sigmoid())

adversary = adversary:cuda()
print(adversary)

--load MNIST data
--[[trainData = mnist.traindataset().data:double():div(255):cuda()
trainlabels = mnist.traindataset().label:cuda()
N = mnist.traindataset().size

testData = mnist.testdataset().data:double():div(255):cuda()
testlabels = mnist.testdataset().label:cuda()
--]]
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
         
trainData = trainset.data:double():div(255):cuda()
--[[trainData = torch.ByteTensor(trainset.data:size())
print(#trainData)
for i=1,trainData:size()[1] do
  trainData[i] = image.rgb2hsv(trainset.data[i])
end
trainData = trainData:double():div(255):cuda()
--]]
trainlabel=trainset.label:cuda()
N = trainData:size()[1]
testData = testset.data
testlabels = testset.label:cuda()


local theta,gradTheta = autoencoder:getParameters()
local thetaAdv,gradThetaAdv = adversary:getParameters()

local criterion = nn.BCECriterion():cuda()

local x

batchSize = 500
iterations = 25

local feval = function(params)
	if theta~=params then
		theta:copy(params)
	end
	gradTheta:zero()
	gradThetaAdv:zero()
	--print(#x)
	local xHat = autoencoder:forward(x)
	local loss = criterion:forward(xHat,x)
	local gradLoss = criterion:backward(xHat,x)
	autoencoder:backward(x,gradLoss)

	local real = torch.Tensor(batchSize,zSize):bernoulli():typeAs(trainData) -- Real Samples
	local YReal = torch.ones(batchSize):typeAs(trainData) -- labels for real samples
	local YFake = torch.zeros(batchSize):typeAs(trainData) --labels for generated samples

	-- Train adversary to maximise log probability of real samples: max_D log(D(x))
	local pred = adversary:forward(real)
	local realLoss = criterion:forward(pred,YReal)
	local gradRealLoss = criterion:backward(pred,YReal)
	adversary:backward(real,gradRealLoss)

	--Train adversary to minimise log probability of fake samples: max_D log(1 - D(G(x)))
	pred = adversary:forward(encoder.output)
	local fakeLoss = criterion:forward(pred,YFake)
	advLoss = realLoss + fakeLoss
	local gradFakeLoss = criterion:backward(pred,YFake)
	local gradFake = adversary:backward(encoder.output, gradFakeLoss)

	-- Train encoder (generator) to play a minimax game with the adversary (discriminator): min_G max_D log(1 - D(G(x)))
	local minimaxLoss = criterion:forward(pred,YReal) -- Technically use max_G max_D log(D(G(x))) for same fixed point, stronger initial gradients
	loss = loss + minimaxLoss
	local gradMinimaxLoss = criterion:backward(pred,YReal)
	local gradMinimax = adversary:updateGradInput(encoder.output, gradMinimaxLoss)
	encoder:backward(x,gradMinimax)

	return loss, gradTheta
end

local advFeval = function(params)
	if thetaAdv~=params then
		thetaAdv:copy(params)
	end
	return advLoss, gradThetaAdv
end

--Train
print('Training Starting')
local optimParams = {learningRate = 0.1}
local advOptimParams = {learningRate = 0.1}
local _,loss 
local losses, advLosses = {},{}
for epoch=1,iterations do
	collectgarbage()
	print('Epoch '..epoch..'/'..iterations)
	for n=1,N, batchSize do
		collectgarbage()
		x = trainData:narrow(1,n,batchSize)
		_,loss = optim.adam(feval,theta,optimParams)
		losses[#losses + 1] = loss[1]
		_,loss = optim.adam(advFeval,thetaAdv,advOptimParams)
		advLosses[#advLosses + 1] = loss[1]
	end
	local plots={{'Adv AE', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
	plots[2]={'Adversary', torch.linspace(1,#advLosses,#advLosses), torch.Tensor(advLosses), '-'}
	gnuplot.pngfigure('AdvAE/Training_bernoulli_cifar_vggsrnn_hsv.png')
	gnuplot.plot(table.unpack(plots))
	gnuplot.ylabel('Loss')
	gnuplot.xlabel('Batch #')
	gnuplot.plotflush()

	--permute training data
	trainData = trainData:index(1,torch.randperm(trainData:size(1)):long())
  
  
  x = testData:narrow(1,1,50)
  --[[local x_hsv = torch.Tensor(x:size()):typeAs(x)
  for i=1,x:size()[1] do
    x_hsv[i] = image.rgb2hsv(x[i])
  end
  --]]
  x_hsv = x--[[_hsv--]]:double():div(255):cuda()
  local xHat_hsv= autoencoder:forward(x_hsv)
  --[[xHat_hsv = xHat_hsv:mul(255):byte()
  for i=1,50 do
    print(i)
    print(xHat_hsv[i][1]:min(),xHat_hsv[i][1]:max())
    print(xHat_hsv[i][2]:min(),xHat_hsv[i][2]:max())
    print(xHat_hsv[i][3]:min(),xHat_hsv[i][3]:max())
  end
  --]]
  --[[local xHat = torch.Tensor(xHat_hsv:size()):typeAs(xHat_hsv)
  for i=1,xHat_hsv:size()[1] do
    xHat[i] = image.hsv2rgb(xHat_hsv[i])
  end
  --]]

  --print (#x)
  ---print(#xHat)
  --temp=torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat,2,50),2)
  --print (#temp)
  image.save('AdvAE/Reconstructions_bernoulli_cifar_vggsrnn_hsv_temp.png', torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat_hsv,2,50),2))
end

print('Testing')
x = testData:narrow(1,1,50)
--[[local x_hsv = torch.Tensor(x:size()):typeAs(x)
for i=1,x:size()[1] do
  x_hsv[i] = image.rgb2hsv(x[i])
end
--]]
x_hsv = x--[[_hsv--]]:double():div(255):cuda()
local xHat_hsv= autoencoder:forward(x_hsv)
--[[xHat_hsv = xHat_hsv:mul(255):byte()
for i=1,50 do
  print(i)
  print(x_hsv[i][1]:min(),x_hsv[i][1]:min())
  print(x_hsv[i][2]:min(),x_hsv[i][2]:min())
  print(x_hsv[i][3]:min(),x_hsv[i][3]:min())
end
--]]
--[[local xHat = torch.Tensor(xHat_hsv:size()):typeAs(xHat_hsv)
for i=1,xHat_hsv:size()[1] do
  xHat[i] = image.hsv2rgb(xHat_hsv[i])
end
--]]

--print (#x)
---print(#xHat)
--temp=torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat,2,50),2)
--print (#temp)
image.save('AdvAE/Reconstructions_bernoulli_cifar_vggsrnn_hsv.png', torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat_hsv,2,50),2))