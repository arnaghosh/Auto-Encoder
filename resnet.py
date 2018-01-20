import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models,transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import os
import matplotlib.pyplot as plt 
from torch.autograd import Function
from collections import OrderedDict
import torch.nn as nn
import math
zsize = 48
batch_size = 11
iterations =  500
learningRate= 0.0001

import torchvision.models as models
#ResNEt#####################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
	
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

###############################################################



###############################################################
class Encoder(nn.Module):

    def __init__(self, block, layers, num_classes=23):
        self.inplanes = 64
        super (Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)
	#self.fc = nn.Linear(num_classes,16) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
	
        x = self.bn1(x)
        x = self.relu(x)
	
        x = self.maxpool(x)
	
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
encoder = Encoder(Bottleneck, [3, 4, 6, 3])
encoder.load_state_dict(torch.load('/home/deepkliv/Downloads/resnet50-19c8e357.pth'))#,map_location=lambda storage, loc: storage.cuda(1)),strict=False)
#loaded_weights = torch.load('/home/siplab/Saket/resnet18-5c106cde.pth')
#print encoder.layer1[1].conv1.weight.data[0][0]
encoder.fc = nn.Linear(2048, 48)
#for param in encoder.parameters():
#    param.requires_grad = False
encoder=encoder.cuda()
y=torch.rand(1,3,224,224)
x=torch.rand(1,128)
x=Variable(x.cuda())
#print decoder(x)
#y=Variable(y.cuda())
#print("\n")
#encoder(y)
#print encoder(y)
##########################################################################
class Binary(Function):

    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

binary = Binary()
##########################################################################
class Decoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		self.dfc3 = nn.Linear(zsize, 4096)
		self.bn3 = nn.BatchNorm2d(4096)
		self.dfc2 = nn.Linear(4096, 4096)
		self.bn2 = nn.BatchNorm2d(4096)
		self.dfc1 = nn.Linear(4096,256 * 6 * 6)
		self.bn1 = nn.BatchNorm2d(256*6*6)
		self.upsample1=nn.Upsample(scale_factor=2)
		self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
		self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
		self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
		self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
		self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)

	def forward(self,x):#,i1,i2,i3):
		
		x = self.dfc3(x)
		#x = F.relu(x)
		x = F.relu(self.bn3(x))
		
		x = self.dfc2(x)
		x = F.relu(self.bn2(x))
		#x = F.relu(x)
		x = self.dfc1(x)
		x = F.relu(self.bn1(x))
		#x = F.relu(x)
		#print(x.size())
		x = x.view(batch_size,256,6,6)
		#print (x.size())
		x=self.upsample1(x)
		#print x.size()
		x = self.dconv5(x)
		#print x.size()
		x = F.relu(x)
		#print x.size()
		x = F.relu(self.dconv4(x))
		#print x.size()
		x = F.relu(self.dconv3(x))
		#print x.size()		
		x=self.upsample1(x)
		#print x.size()		
		x = self.dconv2(x)
		#print x.size()		
		x = F.relu(x)
		x=self.upsample1(x)
		#print x.size()
		x = self.dconv1(x)
		#print x.size()
		x = F.sigmoid(x)
		#print x
		return x
decoder = Decoder()
##########################################
class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder,self).__init__()
		self.encoder = encoder
		self.binary = Binary()
		self.decoder = Decoder()

	def forward(self,x):
		#x=Encoder(x)
		x = self.encoder(x)
		x = binary.apply(x)
		#print x
		#x,i2,i1 = self.binary(x)
		#x=Variable(x)
		x = self.decoder(x)
		return x

#print Autoencoder()

autoencoder = Autoencoder()
#autoencoder = torch.nn.DataParallel(autoencoder, device_ids=[0, 1, 2])
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier,self).__init__()
		self.L1 = nn.Linear(zsize,64)
		self.L2 = nn.Linear(64,32)
		self.L3 = nn.Linear(32,23)

	def forward(self,x):
		x = F.relu(self.L1(x))
		x = F.relu(self.L2(x))
		x = F.log_softmax(self.L3(x))
		return x


#print Classifier()
classifier = Classifier()
#classifier = torch.nn.DataParallel(classifier, device_ids=[0, 1, 2])
class Classification(nn.Module):
	def __init__(self):
		super(Classification,self).__init__()
		self.encoder = encoder
		self.binary = Binary()
		self.classifier = Classifier()

	def forward(self,x):
		x= self.encoder(x)
		x = binary.apply(x)		
		#x= self.binary(x)
		x = self.classifier(x)
		return x

#print Classification()
classification = Classification()

##########################

if torch.cuda.is_available():
	autoencoder.cuda()
	classification.cuda()
	decoder.cuda()
	encoder.cuda()
	classifier.cuda()
	#data

plt.ion()


use_gpu = torch.cuda.is_available()
if use_gpu:
    pinMem = True # Flag for pinning GPU memory
    print('GPU is available!')
else:
    pinMem = False
net = models.resnet18(pretrained=False)
transform = transforms.Compose(
	[
	transforms.Scale((224,224), interpolation=2),
	transforms.ToTensor(),
	#transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
	#transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
	])
trainset=torchvision.datasets.ImageFolder("/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/dataset/train", transform=transform, target_transform=None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)
testset=torchvision.datasets.ImageFolder("/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/dataset/test", transform=transform, target_transform=None)
testloader = torch.utils.data.DataLoader(testset, shuffle = True , batch_size = batch_size , num_workers = 2)

autoencoder_criterion = nn.MSELoss()
classification_criterion = nn.NLLLoss()

autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr = learningRate)
classification_optimizer = optim.Adam(classification.parameters(), lr = learningRate)
#encoder_optimizer = optim.Adam(Encoder.parameters(), lr = learningRate)
list_a_loss = []
list_c_loss = []

#fig = plt.figure()
for epoch in range(iterations):
	run_loss = 0 
	run_c_loss = 0
	autoencoder.train(True) # For training
	classification.train(True)
	for i,data in enumerate(trainloader):
		#print i
		inputs, labels = data
		inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

		
		autoencoder_optimizer.zero_grad()
		classification_optimizer.zero_grad()
		#print(inputs.size())
		pred = autoencoder(inputs)
		#torchvision.utils.save_image(pred.data[0:8], os.path.join('/home/deepkliv/Saket/AE_Classifier/', 'batch_%d_%d'%((epoch+1)/1,i+1) + '.jpg'))
		a_loss = autoencoder_criterion(pred , inputs)
		a_loss.backward()
		autoencoder_optimizer.step()

		#print("efc3", autoencoder.encoder.fc3.bias.grad)
		
		class_pred = classification(inputs)

		c_loss = classification_criterion(class_pred , labels)
	
		#_,xxpred = torch.max(class_pred.data, 1)
		#print("class_pred")
		#print(xxpred.cpu().numpy())
		c_loss.backward(retain_graph=True)
		classification_optimizer.step()
		#encoder_optimizer.step()
		
		run_loss += a_loss.data[0]
		run_c_loss += c_loss.data[0]
		#print i
		if (i +1) % 2 == 0:
			print('[%d, %5d] Autoencoder loss: %.3f Classification loss: %.3f' % (epoch + 1, i + 1 , run_loss/2 , run_c_loss/2))
			#print('[%d,%5d] Classification loss: %.3f' % (epoch + 1, i + 1, run_c_loss/10))
			run_c_loss = 0.0
			run_loss = 0.0


		decoder_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Decoder/', 'decoder-%d.pkl' %(epoch+1))
		encoder_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Encoder/', 'encoder-%d.pkl' %(epoch+1))
		autoencoder_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Autoencoder/', 'autoencoder-%d.pkl' %(epoch+1))
		classifier_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Classifier/', 'classifier-%d.pkl' %(epoch+1))
		classification_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Classification','classification-%d.pkl' %(epoch+1))
		
		torch.save(decoder.state_dict(), decoder_path)
		torch.save(encoder.state_dict(), encoder_path)
		torch.save(autoencoder.state_dict(), autoencoder_path)
		torch.save(classifier.state_dict(), classifier_path)
		torch.save(classification.state_dict(), classification_path)
		
	if ( epoch+1 )% 1 == 0:
		list_a_loss.append(run_loss/5000)
		list_c_loss.append(run_c_loss/5000)
        
		#plt.plot(range(epoch+1),list_a_loss,'r--',label='autoencoder')
		#plt.plot(range(epoch+1),list_c_loss,'b--',label='classifier')
		#if epoch==0:
			#plt.legend(loc='upper left')
			#plt.xlabel('Epochs')
			#plt.ylabel('Loss')
		#fig.savefig('/home/deepkliv/Saket/loss_plot.png') 
		correct = 0
		total = 0
		print('\n Testing ....')
		autoencoder.train(False) # For training
		classification.train(False)
		for t_i,t_data in enumerate(testloader):
						
			if t_i * batch_size >1000:
				break
			t_inputs,t_labels = t_data
			t_inputs = Variable(t_inputs).cuda()
			t_labels = t_labels.cuda()
			t_outputs = autoencoder(t_inputs)
			c_pred = classification(t_inputs)
			_, predicted = torch.max(c_pred.data, 1)
			#print predicted.type() , t_labels.type()
			total += t_labels.size(0)
			correct += (predicted == t_labels).sum()
			if (epoch + 1)%1 == 0:
				print("saving image")
				test_result_path = os.path.join('/home/deepkliv/Desktop/AE/ram/AE_classifier/fashion/Test_results/', 'batch_%d_%d'%((epoch+1)/1,t_i+1) + '.jpg')
				image_tensor = torch.cat((t_inputs.data[0:8], t_outputs.data[0:8]), 0)
				torchvision.utils.save_image(image_tensor, test_result_path)

		print('Accuracy of the network on the 8000 test images: %d %%' % (100 * correct / total))

print('Finished Training and Testing')


"""
classification_criterion = nn.NLLLoss()
for i,data in enumerate(testloader):
	inputs,labels = data
	inputs,labels = Variable(inputs).cuda(), Variable(labels).cuda()
	test_model = Autoencoder().cuda()
	test_model.load_state_dict(torch.load('/home/deepkliv/Saket/AE_Classifier/Autoencoder/autoencoder-500.pkl',map_location=lambda storage, loc: storage.cuda(1)))
	outputs = test_model(inputs)
	test_result_path = os.path.join('/home/deepkliv/Saket/AE_Classifier/Test_results/', 'batch_%d'%(i+1) + '.jpg')
	image_tensor = torch.cat((inputs.data[0:8], outputs.data[0:8]), 0)
	torchvision.utils.save_image(image_tensor, test_result_path)
"""

