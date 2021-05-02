
import torch
import torch.nn as nn
import torch.nn.functional as F
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(n_output),
            ]
            return block

        self.model = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            *convlayer(192, 128, 3, 2, 1), 
            *convlayer(128, 64, 4, 2, 1), #
            *convlayer(64, 1, 3, 1, 0))

    def forward(self, img):
        img = self.model(img)
        # for layer in self.model:
        #     img = layer(img)
        #     #print(img.size())
        return img


# net = Generator()
# print(net)
# x = torch.Tensor(1, 1, 128, 128)
# y = net.forward(x)
# print(y.shape)