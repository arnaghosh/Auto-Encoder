require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
--require 'cudnn';

Data = torch.load('AdvAE/mnist_dict.t7')
Labels = (mnist.traindataset().label+1)
--print(dict_data:size(), trainlabels:size())
zSize = Data:size(2)

adversary = torch.load('AdvAE/adversary.t7')
adversary:cuda()

print("all data in GPU")


N = Data:size(1)
local theta,gradTheta = adversary:getParameters()
criterion = nn.ClassNLLCriterion():cuda()

local x,y

local feval = function(params)
    if theta~=params then
        theta:copy(params)
    end
    gradTheta:zero()
    out = adversary:forward(x)
    --print(#x,#out,#y)
    local loss = criterion:forward(out,y)
    local gradLoss = criterion:backward(out,y)
    adversary:backward(x,gradLoss)
    return loss, gradTheta
end

batchSize = 250

errorTensor = torch.zeros(zSize+1);
local optimParams = {learningRate = 0.001, learningRateDecay = 0.0001}
local _,loss
local losses = {}
for n=1,N-batchSize, batchSize do
    x = Data:narrow(1,n,batchSize):cuda()
    y = Labels:narrow(1,n,batchSize):cuda()
    --print(y)
    _,loss = optim.adam(feval,theta,optimParams)
    losses[#losses + 1] = loss[1]
    --print('Batch '.. n..' done')
    errorTensor[zSize+1] = errorTensor[zSize+1] + loss[1]*batchSize;
end

for f=1,zSize do
    for n=1,N-batchSize, batchSize do
        x = Data:narrow(1,n,batchSize):cuda()
        x[{{},{f}}]:fill(0)
        --print(f,x[1])
        y = Labels:narrow(1,n,batchSize):cuda()
        --print(y)
        _,loss = optim.adam(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
        --print('Batch '.. n..' done')
        errorTensor[f] = errorTensor[f] + loss[1]*batchSize;
    end
end

torch.save('AdvAE/bits_imp.t7',errorTensor);
