# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from models.experimental.functional_fadnetpp.tt.ttnn_resblock import TtResBlock
import ttnn
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
import torch.nn.functional as F
import math


class ttCUNet:
    def build_corr(img_left, img_right, max_disp=40):
        B, C, H, W = img_left.shape
        volume = img_left.new_zeros([B, max_disp, H, W])
        for i in range(max_disp):
            if i > 0:
                volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :-i]).mean(dim=1)
            else:
                volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

        volume = volume.contiguous()
        return volume

    def output_preprocessing(self, output_tensor, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = torch.reshape(
            output_tensor,
            [
                output_tensor.shape[0],
                output_tensor.shape[1],
                int(math.sqrt(output_tensor.shape[3])),
                int(math.sqrt(output_tensor.shape[3])),
            ],
        )
        output_tensor = torch_to_tt_tensor_rm(output_tensor, device, put_on_device=True)
        return output_tensor

    def input_preprocessing(self, input_tensor, device):
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.reshape(
            input_tensor,
            (input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]),
        )
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
        return input_tensor

    def __init__(self, parameters, resBlock=True) -> None:
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.resBlock = resBlock
        if self.resBlock:
            self.conv_redir = TtResBlock(parameters.conv_redir, 128, 32, stride=1)
            self.conv_dy = parameters.conv_dy
            self.conv_d2 = parameters.conv_d2
            self.conv_dy_1 = parameters.conv_dy_1
            self.conv4 = TtResBlock(parameters.conv4, 128, 256, stride=2)
            self.conv4_1 = TtResBlock(parameters.conv4_1, 256, 256, stride=1)
            self.conv5 = TtResBlock(parameters.conv5, 256, 512, stride=2)
            self.conv5_1 = TtResBlock(parameters.conv5_1, 512, 512, stride=1)
            self.conv6 = TtResBlock(parameters.conv6, 512, 1024, stride=2)
            self.conv6_1 = TtResBlock(parameters.conv6_1, 1024, 1024, stride=1)
        else:
            self.conv_redir = parameters.conv_redir
            self.conv_dy = parameters.conv_dy
            self.conv_d2 = parameters.conv_d2
            self.conv_dy_1 = parameters.conv_dy_1
            self.conv4 = parameters.conv4
            self.conv4_1 = parameters.conv4_1
            self.conv5 = parameters.conv5
            self.conv5_1 = parameters.conv5_1
            self.conv6 = parameters.conv6
            self.conv6_1 = parameters.conv6_1
        self.pred_flow6 = parameters.pred_flow6

        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv5.weight = nn.parameter(parameters.iconv5["weight"])
        self.iconv5.bias = nn.parameter(parameters.iconv5["bias"])
        self.iconv4 = nn.ConvTranspose2d(513, 256, 3, 1, 1)
        self.iconv4.weight = nn.parameter(parameters.iconv4["weight"])
        self.iconv4.bias = nn.parameter(parameters.iconv4["bias"])
        self.iconv3 = nn.ConvTranspose2d(257, 128, 3, 1, 1)
        self.iconv3.weight = nn.parameter(parameters.iconv3["weight"])
        self.iconv3.bias = nn.parameter(parameters.iconv3["bias"])
        self.iconv2 = nn.ConvTranspose2d(129, 64, 3, 1, 1)
        self.iconv2.weight = nn.parameter(parameters.iconv2["weight"])
        self.iconv2.bias = nn.parameter(parameters.iconv2["bias"])
        self.iconv1 = nn.ConvTranspose2d(65, 32, 3, 1, 1)
        self.iconv1.weight = nn.parameter(parameters.iconv1["weight"])
        self.iconv1.bias = nn.parameter(parameters.iconv1["bias"])
        self.iconv0 = nn.ConvTranspose2d(35, 32, 3, 1, 1)
        self.iconv0.weight = nn.parameter(parameters.iconv0["weight"])
        self.iconv0.bias = nn.parameter(parameters.iconv0["bias"])

        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv5.weight = nn.parameter(parameters.upconv5["weight"])
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv4.weight = nn.parameter(parameters.upconv4["weight"])
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv3.weight = nn.parameter(parameters.upconv3["weight"])
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv2.weight = nn.parameter(parameters.upconv2["weight"])
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv1.weight = nn.parameter(parameters.upconv1["weight"])
        self.upconv0 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv0.weight = nn.parameter(parameters.upconv0["weight"])

        self.pred_flow5 = parameters.pred_flow5
        self.pred_flow4 = parameters.pred_flow4
        self.pred_flow3 = parameters.pred_flow3
        self.pred_flow2 = parameters.pred_flow2
        self.pred_flow1 = parameters.pred_flow1
        self.pred_flow0 = parameters.pred_flow0

        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow6to5.weight = nn.parameter(parameters.upflow6to5["weight"])
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow5to4.weight = nn.parameter(parameters.upflow5to4["weight"])
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow4to3.weight = nn.parameter(parameters.upflow4to3["weight"])
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow3to2.weight = nn.parameter(parameters.upflow3to2["weight"])
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow2to1.weight = nn.parameter(parameters.upflow2to1["weight"])
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow1to0.weight = nn.parameter(parameters.upflow1to0["weight"])

    def __call__(self, device, inputs, conv1_l, conv2_l, conv3a_l, corr_volume, get_features=False):
        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim=3)
        img_left = imgs[0]
        # img_right = imgs[1]

        out_corr = self.corr_activation(corr_volume)

        if self.resblock:
            out_conv3a_redir = self.conv_redir(conv3a_l)
            in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

            def get_active_filter(in_channel, out_channel=None):
                if out_channel:
                    return self.conv_dy.weight[:out_channel, :in_channel, :, :]
                else:
                    return self.conv_dy.weight[:, :in_channel, :, :]

            in_channel = in_conv3b.size(1)
            filters = get_active_filter(in_channel).contiguous()

            def get_same_padding(kernel_size):
                if isinstance(kernel_size, tuple):
                    assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
                    p1 = get_same_padding(kernel_size[0])
                    p2 = get_same_padding(kernel_size[1])
                    return p1, p2
                assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
                assert kernel_size % 2 > 0, "kernel size should be odd number"
                return kernel_size // 2

            padding = get_same_padding(self.kernel_size)
            # filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
            y = F.conv2d(in_conv3b, filters, None, self.stride, padding, self.dilation, 1)
            bn_d1 = self.bn_d1(y)
            relu_d1 = self.relu(bn_d1)
            conv_d2 = self.conv_d2(relu_d1)
            bn_d2 = self.bn_d2(conv_d2)

            if self.stride != 1 or self.max_out != self.max_in:

                def get_active_filter(n_channel, out_channel=None):
                    if out_channel:
                        return self.conv_dy_1.weight[:out_channel, :in_channel, :, :]
                    else:
                        return self.conv_dy_1.weight[:, :in_channel, :, :]

                in_channel = in_conv3b.size(1)
                filters = get_active_filter(in_channel).contiguous()

                def get_same_padding(kernel_size):
                    if isinstance(kernel_size, tuple):
                        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
                        p1 = get_same_padding(kernel_size[0])
                        p2 = get_same_padding(kernel_size[1])
                        return p1, p2
                    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
                    assert kernel_size % 2 > 0, "kernel size should be odd number"
                    return kernel_size // 2

                padding = get_same_padding(self.kernel_size)
                # filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
                z = F.conv2d(in_conv3b, filters, None, self.stride, padding, self.dilation, 1)
                bn_dy = self.bn_dy(z)
                bn_dy += bn_d2
                conv3b = bn_dy
            else:
                bn_dy = bn_d2
                conv3b = bn_dy

            conv4a = self.conv4(conv3b)
            conv4b = self.conv4_1(conv4a)
            conv5a = self.conv5(conv4b)
            conv5b = self.conv5_1(conv5a)
            conv6a = self.conv6(conv5b)
            conv6b = self.conv6_1(conv6a)

        else:
            out_conv3a_redir = self.conv_redir(conv3a_l)
            # out_conv3a_redir_lr = self.corr_activation(out_conv3a_redir)
            in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

            conv3b = self.conv3_1(in_conv3b)
            # conv3b = self.corr_activation(conv3b_c)
            conv4a = self.conv4(conv3b)
            # conv4a = self.corr_activation(conv4a_c)
            conv4b = self.conv4_1(conv4a)
            # conv4b = self.corr_activation(conv4b_c)
            conv5a = self.conv5(conv4b)
            # conv5a = self.corr_activation(conv5a_c)
            conv5b = self.conv5_1(conv5a)
            # conv5b = self.corr_activation(conv5b_c)
            conv6a = self.conv6(conv5b)
            # conv6a = self.corr_activation(conv6a_c)
            conv6b = self.conv6_1(conv6a)
            # conv6b = self.corr_activation(conv6b_c)

        pr6 = self.pred_flow6(conv6b)
        upconv5 = self.upconv5(conv6b)
        # upconv5 = self.corr_activation(upconv5_c)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        concat5 = self.output_preprocessing(concat5, device)
        iconv5 = self.iconv5(concat5)
        iconv5 = self.input_preprocessing(iconv5, device)
        pr5 = self.pred_flow5(iconv5)
        pr5 = self.output_preprocessing(pr5, device)
        upconv4 = self.upconv4(iconv5)
        # upconv4 = self.corr_activation(upconv4_c)
        upflow5 = self.upflow5to4(pr5)
        upflow5 = self.input_preprocessing(upflow5, device)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        concat4 = self.output_preprocessing(concat4, device)
        iconv4 = self.iconv4(concat4)
        iconv4 = self.input_preprocessing(iconv4, device)
        pr4 = self.pred_flow4(iconv4)
        pr4 = self.output_preprocessing(pr4, device)
        upconv3 = self.upconv3(iconv4)
        # upconv3 = self.corr_activation(upconv3_c)
        upflow4 = self.upflow4to3(pr4)
        upflow4 = self.input_preprocessing(upflow4, device)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        concat3 = self.output_preprocessing(concat3, device)
        iconv3 = self.iconv3(concat3)
        iconv3 = self.input_preprocessing(iconv3, device)
        pr3 = self.pred_flow3(iconv3)
        pr3 = self.output_preprocessing(pr3, device)
        upconv2 = self.upconv2(iconv3)
        # upconv2 = self.corr_activation(upconv2_c)
        upflow3 = self.upflow3to2(pr3)
        upflow3 = self.input_preprocessing(upflow3, device)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        concat2 = self.output_preprocessing(concat2, device)
        iconv2 = self.iconv2(concat2)
        iconv2 = self.input_preprocessing(iconv2, device)

        pr2 = self.pred_flow2(iconv2)
        pr2 = self.output_preprocessing(pr2, device)
        upconv1 = self.upconv1(iconv2)
        # upconv1 = self.corr_activation(upconv1_c)
        upflow2 = self.upflow2to1(pr2)
        upflow2 = self.input_preprocessing(upflow2, device)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        concat1 = self.output_preprocessing(concat1, device)
        iconv1 = self.iconv1(concat1)
        iconv1 = self.input_preprocessing(iconv1, device)

        pr1 = self.pred_flow1(iconv1)
        pr1 = self.output_preprocessing(pr1, device)
        upconv0 = self.upconv0(iconv1)
        # upconv0 = self.corr_activation(upconv0_c)
        upflow1 = self.upflow1to0(pr1)
        upflow1 = self.input_preprocessing(upflow1, device)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        concat0 = self.output_preprocessing(concat0, device)
        iconv0 = self.iconv0(concat0)
        iconv0 = self.input_preprocessing(iconv0, device)

        # predict flow
        pr0 = self.pred_flow0(iconv0)
        pr0 = self.relu(pr0)

        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)

        # can be chosen outside
        if get_features:
            features = (iconv5, iconv4, iconv3, iconv2, iconv1, iconv0)
            return disps, features
        else:
            return disps
