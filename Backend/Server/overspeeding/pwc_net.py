"""
Modern PWC-Net implementation for optical flow estimation
Improved version based on the original PWC-Net paper by Sun et al., 2018
Compatible with modern PyTorch versions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    """Convolution layer with LeakyReLU activation"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def predict_flow(in_planes):
    """Flow prediction layer"""
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    """Deconvolution layer"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class CorrelationLayer(nn.Module):
    """Correlation layer for PWC-Net"""
    
    def __init__(self, max_displacement=4):
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement
        self.pad_size = max_displacement
        
    def forward(self, feature1, feature2):
        """
        Compute correlation between feature maps
        """
        _, _, h, w = feature1.size()
        
        # Pad feature2
        feature2_padded = F.pad(feature2, [self.pad_size] * 4)
        
        # Initialize correlation volume
        correlation_volume = []
        
        for i in range(-self.max_displacement, self.max_displacement + 1):
            for j in range(-self.max_displacement, self.max_displacement + 1):
                # Extract shifted feature2
                shifted_feature2 = feature2_padded[:, :, 
                                  self.pad_size + i:self.pad_size + i + h,
                                  self.pad_size + j:self.pad_size + j + w]
                
                # Compute correlation
                correlation = torch.mean(feature1 * shifted_feature2, dim=1, keepdim=True)
                correlation_volume.append(correlation)
        
        return torch.cat(correlation_volume, dim=1)


class PWCNet(nn.Module):
    """
    PWC-Net for optical flow estimation
    """
    
    def __init__(self, max_displacement=4):
        super(PWCNet, self).__init__()
        
        # Feature extraction pyramid
        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        
        self.conv6a = conv(128, 196, kernel_size=3, stride=2)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)
        
        # Correlation layer
        self.correlation = CorrelationLayer(max_displacement)
        
        # Flow estimation layers
        nd = (2 * max_displacement + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        
        # Level 6
        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        
        # Level 5
        od = nd + 128 + 4
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        
        # Level 4
        od = nd + 96 + 4
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        
        # Level 3
        od = nd + 64 + 4
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        
        # Level 2
        od = nd + 32 + 4
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        
        # Context network (dilated convolutions)
        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def warp(self, x, flow):
        """
        Warp an image/tensor according to optical flow
        """
        B, _, H, W = x.size()
        
        # Create coordinate grid
        xx = torch.arange(0, W, device=x.device, dtype=x.dtype).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device, dtype=x.dtype).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1)
        
        # Add flow to grid
        vgrid = grid + flow
        
        # Scale grid to [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        
        return output

    def forward(self, im1, im2):
        """
        Forward pass of PWC-Net
        """
        # Feature extraction
        c11 = self.conv1b(self.conv1a(im1))
        c21 = self.conv1b(self.conv1a(im2))

        c12 = self.conv2b(self.conv2a(c11))
        c22 = self.conv2b(self.conv2a(c21))

        c13 = self.conv3b(self.conv3a(c12))
        c23 = self.conv3b(self.conv3a(c22))

        c14 = self.conv4b(self.conv4a(c13))
        c24 = self.conv4b(self.conv4a(c23))

        c15 = self.conv5b(self.conv5a(c14))
        c25 = self.conv5b(self.conv5a(c24))

        c16 = self.conv6b(self.conv6a(c15))
        c26 = self.conv6b(self.conv6a(c25))

        # Level 6 flow estimation
        corr6 = self.correlation(c16, c26)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        # Level 5 flow estimation
        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.correlation(c15, warp5)

        # Ensure spatial dimensions match before concatenation
        if up_flow6.shape[2:] != c15.shape[2:]:
            up_flow6 = F.interpolate(up_flow6, size=c15.shape[2:], mode='bilinear', align_corners=False)
        if up_feat6.shape[2:] != c15.shape[2:]:
            up_feat6 = F.interpolate(up_feat6, size=c15.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        # Level 4 flow estimation
        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.correlation(c14, warp4)

        # Ensure spatial dimensions match before concatenation
        if up_flow5.shape[2:] != c14.shape[2:]:
            up_flow5 = F.interpolate(up_flow5, size=c14.shape[2:], mode='bilinear', align_corners=False)
        if up_feat5.shape[2:] != c14.shape[2:]:
            up_feat5 = F.interpolate(up_feat5, size=c14.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        # Level 3 flow estimation
        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.correlation(c13, warp3)

        # Ensure spatial dimensions match before concatenation
        if up_flow4.shape[2:] != c13.shape[2:]:
            up_flow4 = F.interpolate(up_flow4, size=c13.shape[2:], mode='bilinear', align_corners=False)
        if up_feat4.shape[2:] != c13.shape[2:]:
            up_feat4 = F.interpolate(up_feat4, size=c13.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        # Level 2 flow estimation
        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.correlation(c12, warp2)

        # Ensure spatial dimensions match before concatenation
        if up_flow3.shape[2:] != c12.shape[2:]:
            up_flow3 = F.interpolate(up_flow3, size=c12.shape[2:], mode='bilinear', align_corners=False)
        if up_feat3.shape[2:] != c12.shape[2:]:
            up_feat3 = F.interpolate(up_feat3, size=c12.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        # Context network
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        return flow2


def load_pwc_net(model_path=None):
    """
    Load PWC-Net model
    """
    model = PWCNet()

    if model_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(model_path)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded PWC-Net model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using randomly initialized weights")

    return model
