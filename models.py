import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


# 基础模型
class CNN(nn.Module):
    """
    编写一个卷积神经网络类
    """

    def __init__(self):
        """初始化网络,将网络需要的模块拼凑出来。"""
        super(CNN, self).__init__()
        # 卷积层:
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        # 最大池化处理:
        self.pooling = nn.MaxPool2d(2, 2)
        # 全连接层：
        self.fc1 = nn.Linear(16 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        # 用于存储每个线性层的激活值
        self.activations = {"fc1": None, "fc2": None}
        # 记录样本数，用于计算平均值
        self.sample_count = 0

    def forward(self, x):
        """前馈函数"""
        x = F.relu(self.conv1(x))  # = [b, 6, 28, 28]
        x = self.pooling(x)  # = [b, 6, 14, 14]
        x = F.relu(self.conv2(x))  # = [b, 16, 14, 14]
        x = self.pooling(x)  # = [b, 16, 7, 7]
        x = x.view(x.shape[0], -1)  # = [b, 16 * 7 * 7]
        x = F.relu(self.fc1(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc1"] is None:
                    self.activations["fc1"] = torch.zeros(self.fc1.out_features).to(
                        x.device
                    )
                self.activations["fc1"] += torch.sum(torch.abs(x), dim=0)
        self.sample_count += x.size(0)
        x = self.fc2(x)
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc2"] is None:
                    self.activations["fc2"] = torch.zeros(self.fc2.out_features).to(
                        x.device
                    )
                self.activations["fc2"] += torch.sum(torch.abs(x), dim=0)
        output = F.log_softmax(x, dim=1)
        return output


class Texas(nn.Module):
    def __init__(self, num_classes=100, cfg=None):
        super(Texas, self).__init__()
        self.cfg = None
        # self.features = nn.Sequential(
        #     nn.Linear(6169,1024),
        #     nn.Tanh(),
        #     nn.Linear(1024,512),
        #     nn.Tanh(),
        #     nn.Linear(512,256),
        #     nn.Tanh(),
        #     nn.Linear(256,128),
        #     nn.Tanh(),
        # )
        # self.classifier = nn.Linear(128,num_classes)
        self.fc1 = nn.Linear(6169, 1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        # 用于存储每个线性层的激活值
        self.activations = {
            "fc1": None,
            "fc2": None,
            "fc3": None,
            "fc4": None,
            "fc5": None,
        }
        # 记录样本数，用于计算平均值
        self.sample_count = 0

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc1"] is None:
                    self.activations["fc1"] = torch.zeros(self.fc1.out_features).to(
                        x.device
                    )
                self.activations["fc1"] += torch.sum(torch.abs(x), dim=0)
        self.sample_count += x.size(0)
        x = F.tanh(self.fc2(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc2"] is None:
                    self.activations["fc2"] = torch.zeros(self.fc2.out_features).to(
                        x.device
                    )
                self.activations["fc2"] += torch.sum(torch.abs(x), dim=0)
        x = F.tanh(self.fc3(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc3"] is None:
                    self.activations["fc3"] = torch.zeros(self.fc3.out_features).to(
                        x.device
                    )
                self.activations["fc3"] += torch.sum(torch.abs(x), dim=0)
        x = F.tanh(self.fc4(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc4"] is None:
                    self.activations["fc4"] = torch.zeros(self.fc4.out_features).to(
                        x.device
                    )
                self.activations["fc4"] += torch.sum(torch.abs(x), dim=0)
        x = self.fc5(x)
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc5"] is None:
                    self.activations["fc5"] = torch.zeros(self.fc4.out_features).to(
                        x.device
                    )
                self.activations["fc5"] += torch.sum(torch.abs(x), dim=0)
        return x


class PurchaseClassifier(nn.Module):
    def __init__(self, num_classes=100, cfg=None):
        super(PurchaseClassifier, self).__init__()
        self.cfg = cfg
        # self.features = nn.Sequential(
        #     nn.Linear(600,1024),
        #     nn.Tanh(),
        #     nn.Linear(1024,512),
        #     nn.Tanh(),
        #     nn.Linear(512,256),
        #     nn.Tanh(),
        #     nn.Linear(256,128),
        #     nn.Tanh(),
        # )
        # self.classifier = nn.Linear(128,num_classes)
        self.fc1 = nn.Linear(600, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.activations = {
            "fc1": None,
            "fc2": None,
            "fc3": None,
            "fc4": None,
            "fc5": None,
        }
        self.sample_count = 0

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc1"] is None:
                    self.activations["fc1"] = torch.zeros(self.fc1.out_features).to(
                        x.device
                    )
                self.activations["fc1"] += torch.sum(torch.abs(x), dim=0)
        self.sample_count += x.size(0)
        x = F.tanh(self.fc2(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc2"] is None:
                    self.activations["fc2"] = torch.zeros(self.fc2.out_features).to(
                        x.device
                    )
                self.activations["fc2"] += torch.sum(torch.abs(x), dim=0)
        x = F.tanh(self.fc3(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc3"] is None:
                    self.activations["fc3"] = torch.zeros(self.fc3.out_features).to(
                        x.device
                    )
                self.activations["fc3"] += torch.sum(torch.abs(x), dim=0)
        x = F.tanh(self.fc4(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc4"] is None:
                    self.activations["fc4"] = torch.zeros(self.fc4.out_features).to(
                        x.device
                    )
                self.activations["fc4"] += torch.sum(torch.abs(x), dim=0)
        x = self.fc5(x)
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc5"] is None:
                    self.activations["fc5"] = torch.zeros(self.fc4.out_features).to(
                        x.device
                    )
                self.activations["fc5"] += torch.sum(torch.abs(x), dim=0)
        return x


# 混合裁剪模型
class LeNetBN5Mnist(nn.Module):
    def __init__(self, cfg=None, ks=5):
        super(LeNetBN5Mnist, self).__init__()
        if cfg == None:
            self.cfg = [10, "M", 20, "M"]
        else:
            self.cfg = cfg

        self.ks = ks
        self.main = nn.Sequential()
        self.make_layers(self.cfg, True)

        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(self.cfg[-2] * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)
        self.activations = {"fc1": None, "fc2": None}
        self.sample_count = 0

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        idx_maxpool = 1
        idx_bn = 1
        idx_conv = 1
        idx_relu = 1
        for v in self.cfg:
            if v == "M":
                layers += [
                    (
                        "maxpool{}".format(idx_maxpool),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    )
                ]
                idx_maxpool += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks)
                if batch_norm:
                    layers += [
                        ("conv{}".format(idx_conv), conv2d),
                        ("bn{}".format(idx_bn), nn.BatchNorm2d(v)),
                        ("relu{}".format(idx_relu), nn.ReLU(inplace=True)),
                    ]
                    idx_bn += 1
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1
                in_channels = v

        [self.main.add_module(n, l) for n, l in layers]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.main(x)
        # print(x.shape)
        # x = x.view(-1, self.cfg[-2] * self.ks * self.ks)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc1"] is None:
                    self.activations["fc1"] = torch.zeros(self.fc1.out_features).to(
                        x.device
                    )
                self.activations["fc1"] += torch.sum(torch.abs(x), dim=0)
        self.sample_count += x.size(0)
        x = self.fc2(x)

        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc2"] is None:
                    self.activations["fc2"] = torch.zeros(self.fc2.out_features).to(
                        x.device
                    )
                self.activations["fc2"] += torch.sum(torch.abs(x), dim=0)
        return x


class LeNetBN5Fmnist(nn.Module):
    def __init__(self, cfg=None, ks=5):
        super(LeNetBN5Fmnist, self).__init__()
        if cfg == None:
            self.cfg = [20, "M", 40, "M"]
        else:
            self.cfg = cfg

        self.ks = ks
        self.main = nn.Sequential()
        self.make_layers(self.cfg, True)

        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(self.cfg[-2] * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)
        self.activations = {"fc1": None, "fc2": None}
        self.sample_count = 0

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        idx_maxpool = 1
        idx_bn = 1
        idx_conv = 1
        idx_relu = 1
        for v in self.cfg:
            if v == "M":
                layers += [
                    (
                        "maxpool{}".format(idx_maxpool),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    )
                ]
                idx_maxpool += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks)
                if batch_norm:
                    layers += [
                        ("conv{}".format(idx_conv), conv2d),
                        ("bn{}".format(idx_bn), nn.BatchNorm2d(v)),
                        ("relu{}".format(idx_relu), nn.ReLU(inplace=True)),
                    ]
                    idx_bn += 1
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1
                in_channels = v

        [self.main.add_module(n, l) for n, l in layers]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.main(x)
        # print(x.shape)
        # x = x.view(-1, self.cfg[-2] * self.ks * self.ks)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))

        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc1"] is None:
                    self.activations["fc1"] = torch.zeros(self.fc1.out_features).to(
                        x.device
                    )
                self.activations["fc1"] += torch.sum(torch.abs(x), dim=0)
        self.sample_count += x.size(0)
        x = self.fc2(x)

        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc2"] is None:
                    self.activations["fc2"] = torch.zeros(self.fc2.out_features).to(
                        x.device
                    )
                self.activations["fc2"] += torch.sum(torch.abs(x), dim=0)
        return x


class LeNetBN5Cifar(nn.Module):
    def __init__(self, nclasses=10, cfg=None, ks=5):
        super(LeNetBN5Cifar, self).__init__()
        if cfg == None:
            self.cfg = [32, "M", 64, "M"]
        else:
            self.cfg = cfg

        self.ks = ks

        self.main = nn.Sequential()
        self.make_layers(self.cfg, True)

        self.fc1 = nn.Linear(self.cfg[-2] * 5 * 5, 256)
        self.fc2 = nn.Linear(256, nclasses)
        self.activations = {"fc1": None, "fc2": None}
        self.sample_count = 0

        self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        idx_maxpool = 1
        idx_bn = 1
        idx_conv = 1
        idx_relu = 1
        for v in self.cfg:
            if v == "M":
                layers += [
                    (
                        "maxpool{}".format(idx_maxpool),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    )
                ]
                idx_maxpool += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks)
                if batch_norm:
                    layers += [
                        ("conv{}".format(idx_conv), conv2d),
                        ("bn{}".format(idx_bn), nn.BatchNorm2d(v)),
                        ("relu{}".format(idx_relu), nn.ReLU(inplace=True)),
                    ]
                    idx_bn += 1
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1
                in_channels = v

        [self.main.add_module(n, l) for n, l in layers]

    def forward(self, x):
    # x = self.main.conv1(x)
        x = self.main(x)

        # print(x.shape)
        # print(self.cfg[2])
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc1"] is None:
                    self.activations["fc1"] = torch.zeros(self.fc1.out_features).to(
                        x.device
                    )
                self.activations["fc1"] += torch.sum(torch.abs(x), dim=0)
        self.sample_count += x.size(0)
        x = self.fc2(x)

        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc2"] is None:
                    self.activations["fc2"] = torch.zeros(self.fc2.out_features).to(
                        x.device
                    )
                self.activations["fc2"] += torch.sum(torch.abs(x), dim=0)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        return


class LeNetBN5Cifar100(nn.Module):
    def __init__(self, nclasses=10, cfg=None, ks=5):
        super(LeNetBN5Cifar100, self).__init__()
        if cfg == None:
            self.cfg = [64, "M", 128, "M", 256, "M", 512, "M"]
        else:
            self.cfg = cfg

        self.ks = ks
        self.main = nn.Sequential()
        self.make_layers(self.cfg, True)

        self.fc1 = nn.Linear(self.cfg[-2] * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, nclasses)
        self.activations = {"fc1": None, "fc2": None}
        self.sample_count = 0

        self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        idx_maxpool = 1
        idx_bn = 1
        idx_conv = 1
        idx_relu = 1
        for v in self.cfg:
            if v == "M":
                layers += [
                    (
                        "maxpool{}".format(idx_maxpool),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    )
                ]
                idx_maxpool += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks, padding=1)
                if batch_norm:
                    layers += [
                        ("conv{}".format(idx_conv), conv2d),
                        ("bn{}".format(idx_bn), nn.BatchNorm2d(v)),
                        ("relu{}".format(idx_relu), nn.ReLU(inplace=True)),
                    ]
                    idx_bn += 1
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1
                in_channels = v

        [self.main.add_module(n, l) for n, l in layers]

    def forward(self, x):
        # x = self.main.conv1(x)
        x = self.main(x)

        # print(x.shape)
        # print(self.cfg[2])
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc1"] is None:
                    self.activations["fc1"] = torch.zeros(self.fc1.out_features).to(
                        x.device
                    )
                self.activations["fc1"] += torch.sum(torch.abs(x), dim=0)
        self.sample_count += x.size(0)
        x = self.fc2(x)

        if self.training:
            with torch.no_grad():
                # 累加每个神经元的激活值
                if self.activations["fc2"] is None:
                    self.activations["fc2"] = torch.zeros(self.fc2.out_features).to(
                        x.device
                    )
                self.activations["fc2"] += torch.sum(torch.abs(x), dim=0)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        return


def init_net(args):
    users_model = []
    if args.model == "lenet5" and args.dataset == "texas100":
        torch.manual_seed(0)
        net_glob = Texas(num_classes=100, cfg=None).to(args.device)
        for _ in range(args.n_clients):
            torch.manual_seed(0)
            users_model.append(Texas(num_classes=100, cfg=None).to(args.device))
        data_size = None
    if args.model == "lenet5" and args.dataset == "purchase100":
        torch.manual_seed(0)
        net_glob = PurchaseClassifier(num_classes=100, cfg=None).to(args.device)
        for _ in range(args.n_clients):
            torch.manual_seed(0)
            users_model.append(
                PurchaseClassifier(num_classes=100, cfg=None).to(args.device)
            )
        data_size = None
    elif args.model == "lenet5" and (
        args.dataset == "cifar10" or args.dataset == "svhn"
    ):
        torch.manual_seed(0)
        net_glob = LeNetBN5Cifar(nclasses=10, cfg=None, ks=args.ks).to(args.device)
        for _ in range(args.n_clients):
            torch.manual_seed(0)
            users_model.append(
                LeNetBN5Cifar(nclasses=10, cfg=None, ks=args.ks).to(args.device)
            )
        data_size = 5
    elif args.model == "lenet5" and args.dataset == "cifar100":
        torch.manual_seed(0)
        net_glob = LeNetBN5Cifar100(nclasses=100, cfg=None, ks=args.ks).to(args.device)
        for _ in range(args.n_clients):
            torch.manual_seed(0)
            users_model.append(
                LeNetBN5Cifar100(nclasses=100, cfg=None, ks=args.ks).to(args.device)
            )
        data_size = 2
    elif args.model == "lenet5" and (args.dataset == "mnist"):
        torch.manual_seed(0)
        net_glob = LeNetBN5Mnist(cfg=None, ks=args.ks).to(args.device)
        for _ in range(args.n_clients):
            torch.manual_seed(0)
            users_model.append(LeNetBN5Mnist(cfg=None, ks=args.ks).to(args.device))
        data_size = 4
    elif args.model == "lenet5" and (args.dataset == "fmnist"):
        torch.manual_seed(0)
        net_glob = LeNetBN5Fmnist(cfg=None, ks=args.ks).to(args.device)
        for _ in range(args.n_clients):
            torch.manual_seed(0)
            users_model.append(LeNetBN5Fmnist(cfg=None, ks=args.ks).to(args.device))
        data_size = 4
    return users_model, net_glob, data_size


# # 检查模型参数是否相同
# for param1, param2 in zip(model1.parameters(), model2.parameters()):
#     print(torch.equal(param1, param2))  # 所有输出应为 True
