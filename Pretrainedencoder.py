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

zsize = 128
batch_size = 250
iterations =  500
learningRate=0.0001

'''Encoder = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(Encoder.classifier.children())[:-1])
new_classifier.add_module('fc',nn.Linear(4096,zsize))
#new_classifier.add_module('softmax',nn.LogSoftmax())
Encoder.classifier = new_classifier
'''
"""
class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 1000),
			)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x
"""
#new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
#autoencoder network class Encoder(nn.Module):

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder,self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 11, stride = 4, padding = 2)
		self.conv2 = nn.Conv2d(64, 192, 5, padding = 2)
		self.conv3 = nn.Conv2d(192, 384, 3, padding = 1)
		self.conv4 = nn.Conv2d(384, 256, 3, padding = 1)
		self.conv5 = nn.Conv2d(256, 256, 3, padding = 1)
		self.fc1 = nn.Linear(256 * 6 * 6, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, zsize)
#[u'features.0.weight', u'features.3.weight', u'features.6.weight', u'features.8.weight', u'features.10.bias', u'classifier.1.bias', u'classifier.4.bias', u'classifier.6.bias']
	def forward(self,x):
		x = F.relu(self.conv1(x))
		x,indices1 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		
		x = F.relu(self.conv2(x))
		x,indices2 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
		
		x = F.relu(self.conv3(x))
		
		x = F.relu(self.conv4(x))
		
		x = F.relu(self.conv5(x))
		x,indices3 = F.max_pool2d(x,(3,3),(2,2),return_indices = True)
			
		x = x.view(x.size(0), 256 * 6 * 6)
		x = F.dropout(x)
		
		x = F.relu(self.fc1(x))
		x = F.dropout(x)
		
		x = F.relu(self.fc2(x))
		
		x = self.fc3(x)
		
		return x,indices1,indices2,indices3
encoder = Encoder()


#encoder_keys = encoder.state_dict().keys()
#print "Before adding weights", encoder.conv1.weight.data[0,0,0,0]#state_dict()[encoder_keys[0]][63][0]
#print "Before adding bias", encoder.conv1.bias.data[0]
#encoder.load_state_dict(torch.load('/home/siplab/Saket/alexnet-owt-4df8aa71.pth'),strict=False)#,map_location=lambda storage, loc: storage.cuda(1)),strict=False)
#print "After adding weights", encoder.conv1.weight.data[0,0,0,0]#encoder.state_dict()[0]#[encoder_keys[0]][63][0]
loaded_weights = torch.load('/home/deepkliv/Desktop/AE/ram/AE_classifier/alexnet-owt-4df8aa71.pth')
#print loaded_weights['features.0.weight'][0,0,0,0]
#print loaded_weights['features.0.bias'][0]

encoder.conv1.weight = loaded_weights['features.0.weight']
encoder.conv1.bias = loaded_weights['features.0.bias']

encoder.conv2.weight = loaded_weights['features.3.weight']
encoder.conv2.bias = loaded_weights['features.3.bias']

encoder.conv3.weight = loaded_weights['features.6.weight']
encoder.conv3.bias = loaded_weights['features.6.bias']

encoder.conv4.weight = loaded_weights['features.8.weight']
encoder.conv4.bias = loaded_weights['features.8.bias']

encoder.conv5.weight = loaded_weights['features.10.weight']
encoder.conv5.bias = loaded_weights['features.10.bias']

encoder.fc1.weight = loaded_weights['classifier.1.weight']
encoder.fc1.bias = loaded_weights['classifier.1.bias']

encoder.fc2.weight = loaded_weights['classifier.4.weight']
encoder.fc2.bias = loaded_weights['classifier.4.bias']

#encoder = torch.nn.DataParallel(encoder, device_ids=[0, 1, 2])

#encoder.fc3.weight = loaded_weights['classifier.6.weight']
#encoder.fc3.bias = loaded_weights['classifier.6.bias']

#print "After adding weights", encoder.conv1.weight.data[0,0,0,0]#encoder.state_dict()[0]#[encoder_keys[0]][63][0]
#print "After adding bias", encoder.conv1.bias.data[0]
'''class Binary(nn.Module):
	def __init__(self):
		super(Binary,self).__init__()
		self.encoder = Encoder()

	def forward(self, x):
		x,i2,i1 = self.encoder(x)
		x = torch.sign(x)
		#print "x grad ", x.grad
		return x, i2, i1


	def backward(self, grad_output):
		return grad_output'''
class Binary(Function):

    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

binary = Binary()
#binary = torch.nn.DataParallel(binary, device_ids=[0, 1, 2])

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		self.dfc3 = nn.Linear(zsize, 4096)
		self.bn3 = nn.BatchNorm2d(4096)
		self.dfc2 = nn.Linear(4096, 4096)
		self.bn2 = nn.BatchNorm2d(4096)
		self.dfc1 = nn.Linear(4096,256 * 6 * 6)
		self.bn1 = nn.BatchNorm2d(256*6*6)
		self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 1)
		self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
		self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
		self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
		self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 2)

	def forward(self,x,i1,i2,i3):
		
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
		#print x
		x = self.dconv5(F.max_unpool2d(x,i3,kernel_size =(3,3),stride =(2,2)))
		x = F.relu(x)
		
		x = F.relu(self.dconv4(x))
		
		x = F.relu(self.dconv3(x))
		
		x = self.dconv2(F.max_unpool2d(x,i2,kernel_size = (3,3), stride = (2,2)))
		x = F.relu(x)
		
		x = self.dconv1(F.max_unpool2d(x,i1,kernel_size = 3 ,stride = 2))
		#print x
		x = F.sigmoid(x)
		#print x
		return x
decoder = Decoder()
#decoder = torch.nn.DataParallel(decoder, device_ids=[0, 1, 2])
'''
decoder.dconv1.weight = loaded_weights['features.0.weight']
decoder.dconv1.bias = loaded_weights['features.0.bias']

decoder.dconv2.weight = loaded_weights['features.3.weight']
decoder.dconv2.bias = loaded_weights['features.3.bias']

decoder.dconv3.weight = loaded_weights['features.6.weight']
decoder.dconv3.bias = loaded_weights['features.6.bias']

decoder.dconv4.weight = loaded_weights['features.8.weight']
decoder.dconv4.bias = loaded_weights['features.8.bias']

decoder.dconv5.weight = loaded_weights['features.10.weight']
decoder.dconv5.bias = loaded_weights['features.10.bias']

decoder.dfc1.weight = loaded_weights['classifier.1.weight']
decoder.dfc1.bias = loaded_weights['classifier.1.bias']

decoder.dfc2.weight = loaded_weights['classifier.4.weight']
decoder.dfc2.bias = loaded_weights['classifier.4.bias']
'''
class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder,self).__init__()
		self.encoder = Encoder()
		self.binary = Binary()
		self.decoder = Decoder()

	def forward(self,x):
		#x=Encoder(x)
		x,i1,i2,i3 = self.encoder(x)
		x = binary.apply(x)
		#print x
		#x,i2,i1 = self.binary(x)
		#x=Variable(x)
		x = self.decoder(x,i1,i2,i3)
		return x

print Autoencoder()

autoencoder = Autoencoder()
#autoencoder = torch.nn.DataParallel(autoencoder, device_ids=[0, 1, 2])
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier,self).__init__()
		self.L1 = nn.Linear(zsize,64)
		self.L2 = nn.Linear(64,16)
		self.L3 = nn.Linear(16,10)

	def forward(self,x):
		x = F.relu(self.L1(x))
		x = F.relu(self.L2(x))
		x = F.log_softmax(self.L3(x))
		return x


print Classifier()
classifier = Classifier()
#classifier = torch.nn.DataParallel(classifier, device_ids=[0, 1, 2])
class Classification(nn.Module):
	def __init__(self):
		super(Classification,self).__init__()
		self.encoder = Encoder()
		#self.binary = Binary()
		self.classifier = Classifier()

	def forward(self,x):
		x,_,_,_= self.encoder(x)
		#x,_,_ = self.binary(x)
		x = self.classifier(x)
		return x

print Classification()
classification = Classification()


if torch.cuda.is_available():
	autoencoder.cuda()
	classification.cuda()
	decoder.cuda()
	encoder.cuda()
	classifier.cuda()
	#binary.cuda()
#data

plt.ion()

transform = transforms.Compose(
	[
	transforms.Scale((224,224), interpolation=2),
	transforms.ToTensor(),
	#transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
	#transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
	])

trainset = torchvision.datasets.CIFAR10('CIFAR10',train = True , download = True , transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)
testset = torchvision.datasets.CIFAR10('CIFAR10', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, shuffle = False , batch_size = batch_size ,num_workers = 2)
'''trainset=torchvision.datasets.STL10('home/siplab/Saket/STL10', split = 'train' , download = True , transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, shuffle = True , batch_size = batch_size , num_workers = 2)
testset = torchvision.datasets.STL10('home/siplab/Saket/STL10', split = 'test', download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, shuffle = False , batch_size = batch_size ,num_workers = 2)
#classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')'''
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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

	for i,data in enumerate(trainloader):
		#print i
		inputs, labels = data
		inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
		
		autoencoder_optimizer.zero_grad()
		classification_optimizer.zero_grad()
		#print(inputs.size())
		pred = autoencoder(inputs)
		#torchvision.utils.save_image(pred.data[0:8], os.path.join('/home/siplab/Saket/AE_Classifier/', 'batch_%d_%d'%((epoch+1)/1,i+1) + '.jpg'))
		a_loss = autoencoder_criterion(pred , inputs)
		a_loss.backward()
		autoencoder_optimizer.step()

		#print("efc3", autoencoder.encoder.fc3.bias.grad)
		
		class_pred = classification(inputs)
		c_loss = classification_criterion(class_pred , labels)
		#_,xxpred = torch.max(class_pred.data, 1)
		#print("class_pred")
		#print(xxpred.cpu().numpy())
		c_loss.backward()
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


		decoder_path = os.path.join('Decoder/', 'decoder-%d.pkl' %(epoch+1))
		encoder_path = os.path.join('Encoder/', 'encoder-%d.pkl' %(epoch+1))
		autoencoder_path = os.path.join('Autoencoder/', 'autoencoder-%d.pkl' %(epoch+1))
		classifier_path = os.path.join('Classifier/', 'classifier-%d.pkl' %(epoch+1))
		classification_path = os.path.join('Classification/','classification-%d.pkl' %(epoch+1))
		
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
		#fig.savefig('/home/siplab/Saket/loss_plot.png') 
		correct = 0
		total = 0
		print('\n Testing ....')
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
				test_result_path = os.path.join('', 'batch_%d_%d'%((epoch+1)/1,t_i+1) + '.jpg')
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
	test_model.load_state_dict(torch.load('/home/siplab/Saket/AE_Classifier/Autoencoder/autoencoder-500.pkl',map_location=lambda storage, loc: storage.cuda(1)))
	outputs = test_model(inputs)
	test_result_path = os.path.join('/home/siplab/Saket/AE_Classifier/Test_results/', 'batch_%d'%(i+1) + '.jpg')
	image_tensor = torch.cat((inputs.data[0:8], outputs.data[0:8]), 0)
	torchvision.utils.save_image(image_tensor, test_result_path)
"""
