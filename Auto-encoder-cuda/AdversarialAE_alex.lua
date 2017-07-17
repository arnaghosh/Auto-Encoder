require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'cudnn';
torch.manualSeed(1) 
local Threshold,parent = torch.class('nn.Threshold','nn.Module')

function Threshold:__init(th,v, ip)
  parent.__init(self)
  self.threshold = th or 0
  self.val = v or 0
  if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
    error('nn.Threshold(threshold, value)')
  end
  self.inplace = ip or false
  if (ip and type(ip) ~= 'boolean') then
    error('in-place flag must be boolean')
  end
  self:validateParameters()
end

function Threshold:updateOutput(input)
 self:validateParameters()
 input.THNN.Threshold_updateOutput(
    input:cdata(),
    self.output:cdata(),
    self.threshold,
    self.val,
    self.inplace
 )
 return self.output
end

function Threshold:updateGradInput(input, gradOutput)
 self:validateParameters()
 input.THNN.Threshold_updateGradInput(
    input:cdata(),
    gradOutput:cdata(),
    self.gradInput:cdata(),
    self.threshold,
    self.val,
    self.inplace
 )
 return self.gradInput
end

function Threshold:validateParameters()
 self.inplace = self.inplace or false -- backwards compatibility pre inplace
 if self.inplace then
    if self.val > self.threshold then
       error('in-place processing requires value (' .. self.val ..
                ') not exceed threshold (' .. self.threshold .. ')')
    end
 end
end

zSize = 128
--encoder
encoder = nn.Sequential();
encoder:add(nn.SpatialConvolution(3,96,3,3,1,1,1,1)) --1
encoder:add(nn.ReLU()) --2
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --3

encoder:add(nn.SpatialConvolution(96,256,3,3,1,1,1,1)) --4
encoder:add(nn.ReLU()) --5
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --6

encoder:add(nn.SpatialConvolution(256,384,3,3,1,1,1,1)) --7
encoder:add(nn.ReLU()) --8
encoder:add(nn.SpatialConvolution(384,384,3,3,1,1,1,1)) --9
encoder:add(nn.ReLU()) --10
encoder:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1)) --11
encoder:add(nn.ReLU()) --12
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --13

encoder:add(nn.View(-1,4096)) --14
encoder:add(nn.Linear(4096,1024)) --15
--encoder:add(nn.BatchNormalization(1024)) --16
encoder:add(nn.ReLU()) --17
encoder:add(nn.Linear(1024,512)) --18
--encoder:add(nn.BatchNormalization(512)) --19
encoder:add(nn.ReLU()) --20
encoder:add(nn.Linear(512,zSize)) --21
--encoder:add(nn.Sigmoid()) --22

--decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(zSize, 512)) --1
--decoder:add(nn.BatchNormalization(512)) --2
decoder:add(nn.ReLU()) --3
decoder:add(nn.Linear(512, 1024)) --4
--decoder:add(nn.BatchNormalization(1024)) --5
decoder:add(nn.ReLU()) --6
decoder:add(nn.Linear(1024, 4096)) --7
--decoder:add(nn.BatchNormalization(4096)) --8
decoder:add(nn.ReLU()) --9
decoder:add(nn.View(256,4,4)) --10

decoder:add(nn.SpatialMaxUnpooling(encoder:get(13))) --11
decoder:add(nn.SpatialConvolution(256,384,3,3,1,1,1,1)) --12
decoder:add(nn.ReLU()) --13
decoder:add(nn.SpatialConvolution(384,384,3,3,1,1,1,1)) --14
decoder:add(nn.ReLU()) --15
decoder:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1)) --16
decoder:add(nn.ReLU()) --17

decoder:add(nn.SpatialMaxUnpooling(encoder:get(6))) --18
decoder:add(nn.SpatialConvolution(256,96,3,3,1,1,1,1)) --19
decoder:add(nn.ReLU()) --20

decoder:add(nn.SpatialMaxUnpooling(encoder:get(3))) --21
decoder:add(nn.SpatialConvolution(96,3,3,3,1,1,1,1)) --22
decoder:add(nn.Sigmoid()) --23


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
         
--trainData = trainset.data:double():div(255):cuda()
trainData = torch.DoubleTensor(trainset.data:size())
print(#trainData)
for i=1,trainData:size()[1] do
  trainData[i] = image.rgb2hsv(trainset.data[i]:double():div(255))
end

trainData = trainData:cuda()

trainlabel=trainset.label:cuda()
N = trainData:size()[1]
testData = testset.data
testlabels = testset.label:cuda()


local theta,gradTheta = autoencoder:getParameters()
local thetaAdv,gradThetaAdv = adversary:getParameters()

local criterion = nn.MSECriterion():cuda()

local x

batchSize = 512
iterations = 250

local feval = function(params)
	if theta~=params then
		theta:copy(params)
	end
	gradTheta:zero()
	gradThetaAdv:zero()
  --print("x")
	--print(x[1][1])
	local xHat = autoencoder:forward(x)
  --print("xHat")
  --print(xHat[1][1])
	local loss = criterion:forward(xHat,x)
  --print(loss)
	local gradLoss = criterion:backward(xHat,x)
	autoencoder:backward(x,gradLoss)

	local real = torch.Tensor(batchSize,zSize)--[[:bernoulli()--]]:normal(0,1):typeAs(trainData) -- Real Samples
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
local optimParams = {learningRate = 0.002}
local advOptimParams = {learningRate = 0.002}
local _,loss 
local losses, advLosses = {},{}
for epoch=1,iterations do
	collectgarbage()
	print('Epoch '..epoch..'/'..iterations)
	for n=1,N, batchSize do
    if n+batchSize-1>N then break end
		collectgarbage()
		x = trainData:narrow(1,n,batchSize)
		_,loss = optim.adam(feval,theta,optimParams)
		losses[#losses + 1] = loss[1]
		_,loss = optim.adam(advFeval,thetaAdv,advOptimParams)
		advLosses[#advLosses + 1] = loss[1]
	end
	local plots={{'Adv AE', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
	plots[2]={'Adversary', torch.linspace(1,#advLosses,#advLosses), torch.Tensor(advLosses), '-'}
	gnuplot.pngfigure('AdvAE/Training_normal_cifar_hsv_alex.png')
	gnuplot.plot(table.unpack(plots))
	gnuplot.ylabel('Loss')
	gnuplot.xlabel('Batch #')
	gnuplot.plotflush()

	--permute training data
	trainData = trainData:index(1,torch.randperm(trainData:size(1)):long())
  
  
  local x1 = trainData:narrow(1,1,50)
  local xHat= autoencoder:forward(x1)
  --print(x[1])
  --print("xHat111")
  --print(xHat[1][3])
  --[[xHat_hsv = xHat_hsv:mul(255):byte()
  for i=1,50 do
    print(i)
    print(xHat_hsv[i][1]:min(),xHat_hsv[i][1]:max())
    print(xHat_hsv[i][2]:min(),xHat_hsv[i][2]:max())
    print(xHat_hsv[i][3]:min(),xHat_hsv[i][3]:max())
  end
  --]]
  --[[local xHat = torch.DoubleTensor(xHat_hsv:size())
  for i=1,xHat_hsv:size()[1] do
    xHat[i] = image.hsv2rgb(xHat_hsv[i]:double())
  end--]]

  --print (#x)
  ---print(#xHat)
  --temp=torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat,2,50),2)
  --print (#temp)
  image.save('AdvAE/Reconstructions_normal_cifar_hsv_alex_temp.png', torch.cat(image.toDisplayTensor(x1,2,50),image.toDisplayTensor(xHat,2,50),2))
end

torch.save('AdvAE/autoencoder_model_normal.t7',autoencoder)
torch.save('AdvAE/adversary_model_normal.t7',adversary)
print('Testing')
x = testData:narrow(1,1,50):double():div(255):cuda()
local x_hsv = torch.Tensor(x:size()):typeAs(x)
for i=1,x:size()[1] do
  x_hsv[i] = image.rgb2hsv(x[i]:double()) -- It showed an error in this line since the typecasting from CUDA to TorchDouble tensor was not written (DS / 7:20 PM @ 6 Jul)
end
x_hsv = x_hsv:cuda()
local xHat_hsv= autoencoder:forward(x_hsv)
--print(x[1])
--print("xHat111")
--print(xHat[1][1])
--[[xHat_hsv = xHat_hsv:mul(255):byte()
for i=1,50 do
  print(i)
  print(xHat_hsv[i][1]:min(),xHat_hsv[i][1]:max())
  print(xHat_hsv[i][2]:min(),xHat_hsv[i][2]:max())
  print(xHat_hsv[i][3]:min(),xHat_hsv[i][3]:max())
end
--]]
local xHat = torch.DoubleTensor(xHat_hsv:size())
for i=1,xHat_hsv:size()[1] do
  xHat[i] = image.hsv2rgb(xHat_hsv[i]:double())
end
--print (#x)
---print(#xHat)
--temp=torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat,2,50),2)
--print (#temp)
image.save('AdvAE/Reconstructions_normal_cifar_hsv_alex.png', torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat,2,50),2))