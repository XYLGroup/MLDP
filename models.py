import torch.nn as nn
import torch.nn.functional as F
import torch


####### 权值初始化 #######
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


####### 残差块 #######
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


####### 注意力机制层 #######
class SELayer(nn.Module):
    def __init__(self, in_channel=512, reduction=4):
        """ SE注意力机制,输入x。输入输出特征图不变
            1.squeeze: 全局池化 (batch,channel,height,width) -> (batch,channel,1,1) ==> (batch,channel)
            2.excitaton: 全连接or卷积核为1的卷积(batch,channel)->(batch,channel//reduction)-> (batch,channel) ==> (batch,channel,1,1) 输出y
            3.scale: 完成对通道维度上原始特征的标定 y = x*y 输出维度和输入维度相同

        :param channel: 输入特征图的通道数
        :param reduction: 特征图通道的降低倍数
        """
        super(SELayer, self).__init__()
        # 自适应全局平均池化,即，每个通道进行平均池化，使输出特征图长宽为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接的excitation
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid()
        )
        # 卷积网络的excitation
        # 特征图变化：
        # (1,512,1,1) -> (1,512,1,1) -> (1,512,1,1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (batch,channel,height,width) (1,512,32,32)
        b, c, _, _ = x.size()
        # 全局平均池化 (1,512,32,32) -> (1,512,1,1) -> (1,512)
        y = self.avg_pool(x) #.view(b, c, 1, 1)
        # (1,512) -> (1,512//reducation) -> (1,512) -> (1,512,1,1)
        y = self.fc2(y).view(b, c, 1, 1)
        # (1,512,32,32)* (1,512,1,1) -> (1,512,32,32)
        pro = x * y
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        """ 通道注意力机制 同最大池化和平均池化两路分别提取信息，后共用一个多层感知机mlp,再将二者结合

        :param in_channel: 输入通道
        :param ratio: 通道降低倍率
        """
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道先降维后恢复到原来的维数
        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        # return x*self.sigmoid(out)

        # 平均池化一支 (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,8,8) -> (2,512,1,1)
        avg = self.avg_pool(x)
        # 多层感知机mlp (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,1,1) -> (2,512/ratio,1,1)
        avg = self.fc1(avg)
        avg = self.relu1(avg)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        avg_out = self.fc2(avg)

        # 最大池化一支
        # (2,512,8,8) -> (2,512,1,1)
        max = self.max_pool(x)
        # 多层感知机
        # (2,512,1,1) -> (2,512/ratio,1,1)
        max = self.fc1(max)
        max = self.relu1(max)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        max_out = self.fc2(max)

        # (2,512,1,1) + (2,512,1,1) -> (2,512,1,1)
        out = avg_out + max_out
        return x * self.sigmoid(out)


####### 基于残差块的反卷积生成器 #######
class GeneratorResNet_T(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet_T, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(3):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # SELayer
        model += [SELayer(in_channel=in_features)]

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        model += [SELayer(in_channel=in_features)]

        # Upsampling
        for _ in range(3):
            out_features //= 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


####### 注意力机制网络的生成器 #######
class GeneratorResNet_At(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet_At, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 32
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


####### 注意机制网络 #######
class Attention_Net(nn.Module):
    def __init__(self, input_shape):
        super(Attention_Net, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        out_features *= 2
        model += [
            nn.Conv2d(in_features, out_features, 3, stride=4),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Residual blocks
        model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, 1, 7),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


####### 基于PatchGAN的判别器 #######
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, 7, 7)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class VGG16(nn.Module):
    def __init__(self, class_num, flating_num):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(flating_num, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, class_num)
        # softmax 1 * 1 * 1000

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        # out = F.softmax(out, dim=1)

        return out


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self, class_num):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, class_num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


