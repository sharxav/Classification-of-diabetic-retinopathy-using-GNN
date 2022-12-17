#Reference Page - https://github.com/endrol/DR_GCN/blob/9ad1929910ed30c3a623c25ba0da0198bd1655f5/dr_gcn/models.py
import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F


#Initializing the GCN
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

#GCN + CNN
class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=128, t=0, adj_file=None,**kwargs):
        super(GCNResnet, self).__init__()
        #Extracting features from the CNN Model
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        # print('in_channel is ',in_channel)
        '''
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        '''
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        #self.gc3 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
    

        _adj = gen_A(num_classes, t, adj_file)
        # print('in init after get_A ',_adj)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.linear = nn.Linear(20, 5)
       

# Graph structure: 2 - graph convolutional layers with LeakyRelu in between them
    def forward(self, feature, inp):
        feature = self.features(feature)
        #print("Features =",feature.shape)
        feature = self.pooling(feature)
        #print("Features after pooling =",feature.shape)
        feature = feature.view(feature.size(0), -1)
        #print("Features again =",feature.shape)
        inp = inp[0]
        adj1 = gen_adj(self.A)
        #print('adj1 matrix is ', adj1.shape)
        adj = gen_adj(self.A).detach()
        #print('check nan 6 ', adj.shape)
        x = self.gc1(inp, adj)          #First GCN layer
        #print('check nan 5 ', x.shape)
        x = self.relu(x)
        #print('check nan 4 ', x.shape)
        x = self.gc2(x, adj)            #Second GCN layer
        #x = self.relu(x)
        #x = self.gc3(x, adj)
        #print('check nan 3 ', x.shape)
        x = x.transpose(0, 1)
        #print('check nan 2 ', x.shape)
        x = torch.matmul(feature, x)   #Combining CNN and GCN output to classify DR
        #print('before linear x.shape is ',x.shape)
        x = self.linear(x) # Linear layer to get 5 stage classification
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]
'''             
class ResNet101(nn.Module):
    def __init__(self, my_pretrained_model):
        super(ResNet101, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(nn.Conv2d(name='conv2_1',in_channels=2048,out_channels=1024,kernel_size=3,stride=2),
                                           nn.Conv2d(name='conv2_2',in_channels=1024,out_channels=512,kernel_size=3,stride=2),
                                           nn.AdaptiveMaxPool2d(name='gp',output_size=1),
                                           nn.Linear(name='fc',in_features=512,out_features=2048))
     
    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        return x
'''

class net(nn.Module):
  def __init__(self):
    super(net,self).__init__()
    self.conv2_1=nn.Conv2d(2048,1024,kernel_size=3,stride=2)
    self.gp=nn.AdaptiveMaxPool2d(1)
    self.fc=nn.Linear(1024,1000)
    
  def forward(self, x):
    out=self.fc(x)
    return out

#Function to call the model
def gcn_resnet101(num_classes, t, pretrained=True, adj_file=None, in_channel=128):
    model = models.resnet101(pretrained=pretrained)
    #my_model=ResNet101(my_pretrained_model=model)
    #net_add=net()
    #my_model = nn.Sequential(model, net_add)
    #print(my_model)
    #print(model)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
