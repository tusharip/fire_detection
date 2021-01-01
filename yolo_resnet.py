import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim





S=7 #final grid
B=2 #bounding boxes per grid cell
C=2 #total classes

E=C+B*5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class  resnet(nn.Module):
    def __init__(self,model=models.resnet50(pretrained=False).to(device)):

        super().__init__()
        self.resnet_model=model
        
        for name,param in  self.resnet_model.named_parameters():
            if name.startswith('layer4'):
                param.requires_grad=False
                
            else:
                param.requires_grad=False

        self.resnet_model.fc = nn.Sequential(
                            nn.Linear(2048,496),
                            nn.Dropout(0.1),
                            nn.LeakyReLU(0.1),
                            nn.Linear(496,S*S*E)
                            
                        ).to(device)
        print( self.resnet_model)

    def forward(self,x):
        x=F.interpolate(x,scale_factor=0.5,recompute_scale_factor=True)
        x=self.resnet_model(x)
        x=x.view(x.shape[0],S,S,E)
        return  x




