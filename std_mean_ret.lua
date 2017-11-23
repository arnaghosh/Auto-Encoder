require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'cudnn';
require './BinarizedNeurons'
local fashion_mnist = require 'fashion-mnist';
ret_vec=torch.load("/home/siplab/AE/AE/fashion_mnist_retrieval_cvpr.t7")
mytable={}
mytable2={}
for j=1,10 do
  x=0
  mytable3={}
  count=1
  y=torch.zeros(1000)
  for i=1,10000 do
  
  if (ret_vec[i][1]==j) 
  then  temp=ret_vec[i][2]  x=x+temp y[count] = temp count=count+1    
  --print(ret_vec[i][1],ret_vec[i][2],ret_vec[i][3],y)
        end 
     
  end
  
  table.insert(mytable,j,torch.std(y))
  table.insert(mytable2,j,x/count)
end
print(mytable,mytable2)
