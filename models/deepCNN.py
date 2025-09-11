import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=1,
                 norm_type='batch',
                 activation='custom'
                ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.out_chans = 1
        
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm([out_channels, 1, 1])
        else:
            self.norm = nn.Identity()
        
        if activation == 'custom':
            self.act = self.custom_act
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()
        
        self.drop = nn.Dropout(0.1)

    def custom_act(self, x):
        return torch.where(
            x <= 0, 0.1 * x,
            torch.where(x >= 10, torch.tensor(10.0, device=x.device), x)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, img_size=[144, 288], in_chans=1, base_chans=32,out_chans=1,
                drop_rate=0., use_checkpoint=False, **kwargs):
        super().__init__()

        self.layer1 = Conv2DBlock(in_channels=in_chans,out_channels=base_chans)
        self.layer2 = Conv2DBlock(in_channels=base_chans,out_channels=base_chans)

        self.layer3 = Conv2DBlock(in_channels=base_chans,out_channels=base_chans*2)
        self.layer4 = Conv2DBlock(in_channels=base_chans*2,out_channels=base_chans*2)

        self.layer5 = Conv2DBlock(in_channels=base_chans*2,out_channels=base_chans*4)
        self.layer6 = Conv2DBlock(in_channels=base_chans*4,out_channels=base_chans*2)

        self.layer7 = Conv2DBlock(in_channels=base_chans*4,out_channels=base_chans*2)
        self.layer8 = Conv2DBlock(in_channels=base_chans*2,out_channels=base_chans*1)

        self.layer9 = Conv2DBlock(in_channels=base_chans*2,out_channels=base_chans)
        self.layer10 = Conv2DBlock(in_channels=base_chans,out_channels=base_chans)
        self.layer11 = Conv2DBlock(in_channels=base_chans,out_channels=out_chans)

        self.avg_pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None):

        # x shape: (B,1,H,W)
        x = self.layer1(x)
        x = self.layer2(x)
        skip0 = x

        x = self.avg_pool_layer(x)
        x = self.layer3(x)
        x = self.layer4(x)
        skip1 = x

        x = self.avg_pool_layer(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.upsample(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.upsample(x)
        x = torch.cat([x, skip0], dim=1)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        return x
