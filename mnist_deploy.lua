require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'cudnn';
require './BinarizedNeurons'

classes ={0,1,2,3,4,5,6,7,8,9}
zSize = 16
--encoder
encoder = torch.load('/home/siplab/AE/mnist/encoder_1.t7');
classifier=torch.load('/home/siplab/AE/mnist/classifier.t7');
binariser = nn.Sequential();
binariser:add(BinarizedNeurons())

autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(binariser)
autoencoder:add(classifier)
autoencoder = autoencoder:cuda()
print(autoencoder)

--load MNIST data
trainData = mnist.traindataset().data:double():div(255):reshape(60000,1,28,28):cuda()
trainlabels = (mnist.traindataset().label+1):cuda()
N = mnist.traindataset().size

testData = mnist.testdataset().data:double():div(255):reshape(10000,1,28,28):cuda()
testlabels = (mnist.testdataset().label+1):cuda()
teSize = mnist.testdataset().size
print(N,teSize)
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
gt=torch.Tensor(10000,1):cuda()
for i=1,10000 do
  gt[i]=testlabels[i]
end
count=0
 
for n=1,10000 do
	local groundtruth = gt[n]
	x=autoencoder:forward(testData[n])
        confidences, indices = torch.sort(x, true)
	print (indices[1],gt[n])
        if torch.Tensor(1):fill(indices[1]):cuda():equal(groundtruth)  then
        --print(1,indices[1])
	count=count+1
        --print(2,groundtruth)
        class_performance[indices[1]] = class_performance[indices[1]] + 1
    	end
end
end
for i=1,#classes do
    print (class_performance[i])
end
for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end
