"""
CNN model for 32x32x14 multimodal classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sar_conv1 = nn.Conv2d(4, 4, 5, 1)
        self.sar_conv2 = nn.Conv2d(4, 10, 5, 1)

        self.eo_conv1 = nn.Conv2d(10, 4, 5, 1)
        self.eo_conv2 = nn.Conv2d(4, 10, 5, 1)

        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 17)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # from NHWC to NCHW
        sar_x = x[:, :4, :, :]
        eo_x = x[:, 4:, :, :]

        sar_x = self.sar_conv1(sar_x)
        sar_x = F.relu(sar_x)
        sar_x = F.max_pool2d(sar_x, 2)
        sar_x = self.sar_conv2(sar_x)
        sar_x = F.relu(sar_x)
        sar_x = F.max_pool2d(sar_x, 2)
        sar_x = torch.flatten(sar_x, 1)

        eo_x = self.eo_conv1(eo_x)
        eo_x = F.relu(eo_x)
        eo_x = F.max_pool2d(eo_x, 2)
        eo_x = self.eo_conv2(eo_x)
        eo_x = F.relu(eo_x)
        eo_x = F.max_pool2d(eo_x, 2)
        eo_x = torch.flatten(eo_x, 1)

        x = torch.cat((sar_x, eo_x), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def make_so2sat_model(**kwargs):
    return Net()
