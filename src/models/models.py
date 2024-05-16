import torch
from torch import nn
import torchvision
from torchvision.models.convnext import LayerNorm2d
from torchvision.models.convnext import ConvNeXt_Tiny_Weights
#from Efficient3DCNNs.models import resnext

import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspaces/markoc-haeslerlab/convnextv2/ConvNeXt-V2/')
sys.path.insert(0, '/deepsniff/convnextv2/ConvNeXt-V2/')
#from models.convnextv2 import ConvNeXtV2

class MiniConvNet(nn.Module):

    def __init__(self, n_input_channels=1, output_dim=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.fc1 = nn.Linear(32*56*76, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):                               #240x320
        x = torch.nn.functional.relu(self.conv1(x))     #236x316
        x = torch.nn.functional.max_pool2d(x, 3, 2)     #118x158
        x = torch.nn.functional.relu(self.conv2(x))     #114x154
        x = torch.nn.functional.max_pool2d(x, 3, 2)     #56x76
        x = x.view(-1, 32*56*76)                        #32*56*76
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MobileNetV3(nn.Module):

    def __init__(self, n_input_channels=3, output_dim=1, **kwargs):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.output_dim =   output_dim

        self.model = torchvision.models.mobilenet_v3_small(weights=None, progress=True)
        self.model._modules["features"][0][0] = nn.Conv2d(n_input_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(in_features=576, out_features=output_dim, bias=True)
        )
    def forward(self, x):
        x = self.model(x)
        return x


class SniffLSTM(nn.Module):
    '''
    The model input is a sequence of images, and the output is a single value. Images are processed by a pretrained CNN encoder, and the output is fed into a LSTM.
    '''
    def __init__(self, encoder, lstm_input_size=64, hidden_size=32, num_layers=1, output_dim=1, bidirectonal=True, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.lstm = torch.nn.LSTM(lstm_input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectonal)
        
        fc_input_size = hidden_size if not bidirectonal else hidden_size*2
        self.fc = torch.nn.Linear(fc_input_size, output_dim, bias=False, device=None, dtype=None)
    
    def forward(self, x):
        x = x.unsqueeze(2)
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.encoder(x)
        x = x.view(batch_size, sequence_length, -1)
        x, hidden = self.lstm(x)
        x = x[:, sequence_length//2:sequence_length//2 + 1, :]
        x = self.fc(x)
        return x.squeeze(2)




class RegressionResnet(nn.Module):
    #default weights
    def __init__(self, architecture='resnet18', n_input_channels=3, output_dim=1, weights=None, **kwargs):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', architecture, weights=weights)
        self.model.conv1 = torch.nn.Conv2d(n_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if architecture == 'resnet18':
            self.model.fc = torch.nn.Linear(512, output_dim, bias=False, device=None, dtype=None)
        elif architecture == 'resnet50':
            self.model.fc = torch.nn.Linear(2048, output_dim, bias=False, device=None, dtype=None)

    def forward(self, x):
        x = self.model(x)
        return x

class RegressionResnetV2(nn.Module):
    
    def __init__(self, architecture='resnet18', sequence_size=3, input_dim=1, output_dim=1, weights=None, **kwargs):
            super().__init__()
            self.model = torch.hub.load('pytorch/vision:v0.10.0', architecture, weights=weights)
            self.model.conv1 = torch.nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            if architecture == 'resnet18':
                self.model.fc = torch.nn.Linear(512, 64, bias=False, device=None, dtype=None)
            elif architecture == 'resnet50':
                self.model.fc = torch.nn.Linear(2048, output_dim, bias=False, device=None, dtype=None)
            self.fc = torch.nn.Linear(64*sequence_size, output_dim, bias=False, device=None, dtype=None)
    def forward(self, x):
        #Merge first two dimensions of (B, C, H, W) -> (B*C, 1, H, W)
        print(x.shape)
        x = x.view(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        print(x.shape)
        x = self.model(x)
        #go back to (B, C, H, W)
        print(x.shape)
        x = x.view(x.shape[0]//x.shape[1], x.shape[1], x.shape[2], x.shape[3])
        print
        #relu
        x = torch.nn.functional.relu(x)
        #flatten all but batch dimension
        print(x.shape)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = self.fc(x)
        return x

class ConvNext(nn.Module):
    def __init__(self, n_input_channels=3, output_dim=1, weights = None, **kwargs):
        super().__init__()
        self.model = torchvision.models.convnext_tiny(weights = weights, progress = True, num_classes = 1000)
        self.model._modules["features"][0][0] = nn.Conv2d(n_input_channels, 96, kernel_size=(4, 4), stride=(4, 4))
        #change last layer to output_dim    
        
        #freeze all layers except last
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier = nn.Sequential(
            LayerNorm2d((768,), eps=1e-06, elementwise_affine=True), 
            nn.Flatten(1), 
            nn.Linear(768, 10)
        )

        #check if self.model.classifier requires grad
        for param in self.model.parameters():
            print(param.requires_grad)

        print('loaded ConvNext')

    def forward(self, x): 
        x = self.model(x)
        return x

class VGG1(nn.Module):
    def __init__(self, n_input_channels=3, output_dim=1, weights = None, **kwargs):
        super(VGG1, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2240, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x



class CNNModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(5, 8, kernel_size=3, padding=1) #112
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) #56
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   #28
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    #14
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   #7
        self.conv6 = nn.Conv2d(128, 128, kernel_size=7, padding=0)  #1
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 112
        x = self.pool(F.relu(self.conv2(x)))   ## 56
        x = self.pool(F.relu(self.conv3(x)))   ## 28
        x = self.pool(F.relu(self.conv4(x)))   # 14
        x = self.pool(F.relu(self.conv5(x)))   # 7
        x = F.relu(self.conv6(x))   #1
        # Flatten the feature maps
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))

        return x

class ResNet2dplus1d_18(nn.Module):
    def __init__(self, num_input_channels=5, output_dim=1, weights=False, **kwargs):
        super(ResNet2dplus1d_18, self).__init__()
        # Load the pretrained ResNet model
        self.resnet = torchvision.models.video.r2plus1d_18(weights=weights)
        # Change the number of classes
        self.resnet.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        return self.resnet(x)


# def resnext3d101(weights=None):
#     model = resnext.resnext101(sample_size=112, sample_duration=16, num_classes=600)

#     if weights:
#         state_dict = torch.load(weights)['state_dict'] 
#         #remove module. prefix
#         state_dict = {k[7:]: v for k, v in state_dict.items()}
#         model.load_state_dict(state_dict)

#     #change last model layer to output_dim
#     model.fc = torch.nn.Linear(model.fc.in_features, 1)
        
#     #freeze all layers except the last one
#     for param in model.parameters():
#         param.requires_grad = False
#     model.fc.weight.requires_grad = True
#     model.fc.bias.requires_grad = True

#     return model