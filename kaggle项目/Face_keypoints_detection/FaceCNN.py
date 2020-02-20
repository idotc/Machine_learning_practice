import torch
import torch.nn as nn

def xavier_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
        for name, _ in m.named_parameters():
            if name in ['bias']:
                nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  #48
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2), #24
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2), #12
        )
        self.conv1.apply(xavier_normal_init)
        self.conv2.apply(xavier_normal_init)
        self.conv3.apply(xavier_normal_init)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256*12*12, 4096),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.PReLU(),
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.Linear(256, 30),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
