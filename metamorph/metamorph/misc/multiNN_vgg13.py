import torch
import torch.nn as nn

class multiNN_vgg13(nn.Module):
    def __init__(self, num_out):
        super(multiNN_vgg13, self).__init__()
        self.net = nn.Sequential(
            # first batch (64)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # second batch (128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Third Batch (256)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 4-th Batch (512)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 5-th Batch (512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # FC
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_out),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class TridentNN_vgg13(nn.Module):
    def __init__(self, num_age, num_gen, num_eth):
        super(TridentNN_vgg13, self).__init__()
        self.ageNN = multiNN_vgg13(num_out=num_age)
        self.genNN = multiNN_vgg13(num_out=num_gen)
        self.ethNN = multiNN_vgg13(num_out=num_eth)

    def forward(self, x):
        age = self.ageNN(x)
        gen = self.genNN(x)
        eth = self.ethNN(x)

        return age, gen, eth


if __name__ == '__main__':
    print('Testing out Multi task Network')
    mtNN = TridentNN_vgg13(10, 2, 3)
    input = torch.randn(64, 1, 48, 48)
    y = mtNN(input)
    print(mtNN)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)

    print(mtNN.parameters())