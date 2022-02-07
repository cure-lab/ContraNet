import torch.nn as nn
import math
import torch


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=12, input_size=64, width_mult=1.,
                 change_mlp=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1], # change this 2->1 for 32x32 input.
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)
        # first channel is always 32!
        self.last_channel = make_divisible(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        # change this 2->1 for 32x32 input.
        self.features = [conv_bn(3, input_channel, 1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        if change_mlp:
            self.classifier = MLP_normal(self.last_channel)
        else:
            self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()
        self.with_features = False

    def query_features(self, use=True):
        self.with_features = use

    def forward(self, x):
        x = self.features(x)
        features = x.mean(3).mean(2)
        x = self.classifier(features)
        if self.with_features:
            return features, x
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=2, p=0.1, class_num=10):
        super(MLP, self).__init__()
        # in_dim will be 1280x2 for MobileNet due to mean()
        print(">>>> This MLP model takes {} in classes and out 2 <<<<".format(
            class_num))
        self.feature1 = nn.ModuleDict()
        for i in range(class_num):
            self.feature1[str(i)] = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.Dropout(p),
                nn.ReLU()
            )
        self.feature2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, din):
        x = din[0]
        y = din[1]
        dout = []
        for idx, i in enumerate(y):
            label = str(i.item())
            dout.append(self.feature1[label](x[idx].unsqueeze(0)))
        dout = torch.cat(dout, dim=0)
        dout = self.feature2(dout)
        dout = nn.functional.softmax(dout)
        return dout


class MLP_normal(nn.Module):
    def __init__(self, in_dim, out_dim=2, p=0.1):
        super(MLP_normal, self).__init__()
        # in_dim will be 1280x2 for MobileNet due to mean()
        self.feature1 = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Dropout(p),
            nn.ReLU()
        )
        self.feature2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, din):
        dout = self.feature1(din)
        dout = self.feature2(dout)
        dout = nn.functional.softmax(dout)
        return dout


class MobileNetV2f_c(MobileNetV2):
    # This is for fix random parameter apply on the forth channel.
    def __init__(self, n_class=12, input_size=64, width_mult=1.,
                 change_mlp=False, in_channel=3):
        super().__init__(n_class=n_class, input_size=input_size,
                         width_mult=width_mult, change_mlp=change_mlp)
        print(">>>> fix random parameter apply on the forth channel <<<<")
        input_channel = 32
        torch.random.manual_seed(0)
        sampler = torch.distributions.uniform.Uniform(0, 1)
        self.channel_patch = {i: sampler.sample(
            (1, input_size, input_size)) for i in range(10)}
        self.features[0] = conv_bn(in_channel + 1, input_channel, 2)
        for m in self.features[0].modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data):
        x = data[0]
        y = data[1]
        new_x = []
        for i in range(x.size(0)):
            new_x.append(torch.cat(
                [x[i], self.channel_patch[y[i].item()].to(x.device)], dim=0))
        x = torch.stack(new_x)
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


class MobileNetV2f_m(MobileNetV2):
    # This is for fix random parameter multiply all three channels.
    def __init__(self, n_class=12, input_size=64, width_mult=1.):
        super().__init__(n_class=n_class, input_size=input_size,
                         width_mult=width_mult)
        print(">>>> fix random parameter multiply all three channels <<<<")
        torch.random.manual_seed(0)
        sampler = torch.distributions.uniform.Uniform(0, 1)
        self.channel_patch = {i: sampler.sample(
            (1, input_size, input_size)) for i in range(10)}

    def forward(self, data):
        x = data[0]
        y = data[1]
        patch = []
        for i in range(x.size(0)):
            patch.append(self.channel_patch[y[i].item()])
        patch = torch.stack(patch).to(x.device)
        x = x * patch
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


class MobileNetV2l_c(MobileNetV2):
    # This is for learnable random parameter apply on the forth channel.
    def __init__(self, n_class=12, input_size=64, width_mult=1.,
                 change_mlp=False, in_channel=3):
        super().__init__(n_class=n_class, input_size=input_size,
                         width_mult=width_mult, change_mlp=change_mlp)
        print(">>>> learnable random parameter apply on the forth channel <<<<")
        input_channel = 32
        torch.random.manual_seed(0)
        sampler = torch.distributions.uniform.Uniform(0, 1)
        self.channel_patch = {i: torch.nn.Parameter(
            sampler.sample((1, input_size, input_size))
        ) for i in range(10)}
        self.features[0] = conv_bn(in_channel + 1, input_channel, 2)
        for m in self.features[0].modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data):
        x = data[0]
        y = data[1]
        new_x = []
        for i in range(x.size(0)):
            new_x.append(torch.cat(
                [x[i], self.channel_patch[y[i].item()].to(x.device)], dim=0))
        x = torch.stack(new_x)
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


class MobileNetV2l_m(MobileNetV2):
    # This is for learnable random parameter multiply all three channels.
    def __init__(self, n_class=12, input_size=64, width_mult=1.):
        super().__init__(n_class=n_class, input_size=input_size,
                         width_mult=width_mult)
        print(">>>> learnable random parameter multiply all three channels <<<<")
        torch.random.manual_seed(0)
        sampler = torch.distributions.uniform.Uniform(0, 1)
        self.channel_patch = {i: torch.nn.Parameter(
            sampler.sample((1, input_size, input_size))
        ) for i in range(10)}

    def forward(self, data):
        x = data[0]
        y = data[1]
        patch = []
        for i in range(x.size(0)):
            patch.append(self.channel_patch[y[i].item()])
        patch = torch.stack(patch).to(x.device)
        x = x * patch
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


class MobileNetV2fc(MobileNetV2):
    # This is for one-hot vector concat to the feature vector.
    def __init__(self, n_class=12, input_size=64, width_mult=1.):
        super().__init__(n_class=n_class, input_size=input_size,
                         width_mult=width_mult)
        print(">>>> one-hot vector concat to the feature vector <<<<")
        self.classifier = nn.Linear(self.last_channel + 10, n_class)
        # self.classifier = nn.Linear(self.last_channel + 1, n_class)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()

    def forward(self, data):
        x = data[0]
        y = data[1].unsqueeze(1)
        with torch.no_grad():
            y = torch.zeros(
                [y.size(0), 10],
                dtype=torch.float32
            ).to(y.device).scatter_(1, y, 1.)
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = torch.cat([x, y], dim=-1)
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1',
            progress=True)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    net = mobilenet_v2(True)
