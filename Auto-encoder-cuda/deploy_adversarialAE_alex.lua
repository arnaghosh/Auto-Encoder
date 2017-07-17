require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'cudnn';

--define function binarise
function binarise(x,thresh,low,high)
  y = torch.Tensor(x:size())
  for i=1,x:size(2) do
    if x[1][i]<=thresh then
      y[1][i]=low
    else
      y[1][i]=high
    end
  end 
  return y
end

--define function to sort table
function compare(a,b)
  return tostring(a[2])<tostring(b[2])
end

--define xor function for tensors
function xorTensor(A,B)
  C = torch.mod(torch.add(A,B),2)
  return C
end
  
--load model
autoencoder = torch.load('AdvAE/autoencoder_model_sigmoid.t7')
encoder = autoencoder:get(1)

--load data
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

trainData = torch.DoubleTensor(trainset.data:size())
print(#trainData)
for i=1,trainData:size()[1] do
  trainData[i] = image.rgb2hsv(trainset.data[i]:double():div(255))
end

trainData = trainData:cuda()
trainlabel=trainset.label
N = trainData:size(1)

dict={}
for i=1,N do
  z = encoder:forward(trainData[i])
  --print(z:min(),z:max())
  z1 = binarise(z,0.5,0,1)
  dict[i] = {i,trainlabel[i],z1:byte()}
end
print("done building dict")
--print(dict)

--calculate accuracy
top1 = 0
top3 = 0
top5 = 0
for k=1,testset.data:size(1) do
  collectgarbage()
  temp={}
  if k%1000 then print(k,top1,top3,top5) end
  A = image.rgb2hsv(testset.data[k]:double():div(255))
  A = encoder:forward(A:cuda())
  A = binarise(A,0.5,0,1)
  for i=1,N do
    temp[i] = {dict[i][2], xorTensor(dict[i][3],A:byte())}
  end

  table.sort(temp,compare)
  target = testset.label[k]
  --print(target, temp[1][1],temp[2][1],temp[3][1],temp[4][1],temp[5][1])
  if target == temp[1][1] then
    top1 = top1+1
    top3 = top3+1
    top5 = top5+1
  elseif target == temp[2][1] or target == temp[3][1] then
    top3 = top3+1
    top5 = top5+1
  elseif target ==temp[4][1] or target ==temp[5][1] then
    top5 = top5+1
  end
end

--print the accuracies
print('Top 1-recall accuracy ' .. top1*100/testset.data:size(1) .. '%')
print('Top 3-recall accuracy ' .. top3*100/testset.data:size(1) .. '%')
print('Top 5-recall accuracy ' .. top5*100/testset.data:size(1) .. '%')
--[[print(dict[1][3])
print(dict[2][3])
print(dict[3][3])
print(dict[4][3])
print(dict[5][3])
print(testset.label[1])
--]]