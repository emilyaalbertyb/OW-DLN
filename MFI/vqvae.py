import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


# class VQVAE(nn.Module):
#     def __init__(
#         self,
#         in_channel=3,
#         channel=128,
#         n_res_block=2,
#         n_res_channel=32,
#         embed_dim=64,
#         n_embed=512,
#         decay=0.99,
#     ):
#         super().__init__()
#
#         self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
#         self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
#         self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
#         self.quantize_t = Quantize(embed_dim, n_embed)
#         self.dec_t = Decoder(
#             embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
#         )
#         self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
#         self.quantize_b = Quantize(embed_dim, n_embed)
#         self.upsample_t = nn.ConvTranspose2d(
#             embed_dim, embed_dim, 4, stride=2, padding=1
#         )
#         self.dec = Decoder(
#             embed_dim + embed_dim,
#             in_channel,
#             channel,
#             n_res_block,
#             n_res_channel,
#             stride=4,
#         )
#
#
#     def forward(self, input):
#         quant_t, quant_b, diff, _, _ = self.encode(input)
#         dec = self.decode(quant_t, quant_b)
#
#         return dec, diff

class VQVAE(nn.Module):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_embed=512,
            decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        # self.discriminator = nn.Sequential(
        #     nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(16, 1, kernel_size=4, stride=1, padding=0),
        #     nn.Sigmoid()
        # )

        # self.discriminator = nn.Sequential(
        #     nn.Conv2d(in_channel,channel, 4, 2, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(channel, channel * 2, 4, 2, 1),
        #     nn.BatchNorm2d(channel * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(channel * 2, channel * 4, 4, 2, 1),
        #     nn.BatchNorm2d(channel * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(channel * 4, 1, 4, 1, 0),
        #     nn.Sigmoid()
        # )
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channel, channel, 4, 2, 1),  # 输入通道数为 in_channel
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel * 2, 4, 2, 1),
            nn.BatchNorm2d(channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel * 2, channel * 4, 4, 2, 1),
            nn.BatchNorm2d(channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel * 4, 1, 4, 1, 0),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)  # 将 height 和 width 归一化为 1
        )

        self.discriminator_feat = nn.Sequential(
            nn.Conv2d(embed_dim, channel, 4, 2, 1),  # 输入通道数为 embed_dim
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel * 2, 4, 2, 1),
            nn.BatchNorm2d(channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel * 2, channel * 4, 4, 2, 1),
            nn.BatchNorm2d(channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        # Calculate discriminator output
        disc_output = self.discriminator_feat(quant_t)
        return dec, diff, disc_output
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.model(x)
#
    # def forward(self, input):
    #     quant_t, quant_b, diff, _, _ = self.encode(input)
    #     dec = self.decode(quant_t, quant_b)
    #     return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

# class Discriminator(nn.Module):
#     def __init__(self,in_features=784):
#         """真实数据的维度,同时也是生成的假数据的"""
#         super().__init__()
#         self.disc = nn.Sequential(nn.Linear(in_features,128),
#                                  nn.LeakyReLU(0.1), #由于生成对抗网络的损失非常容易梯度消失，因此使用LeakyReLU
#                                  nn.Linear(128,1),
#                                  nn.Sigmoid()
#                                  )
#     def forward(self,data):
#         """输入的data可以是真实数据时，Disc输出dx。输入的data是gz时，Disc输出dgz"""
#         return self.disc(data)


## ##### 定义判别器 Discriminator ######
## 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
## 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类
class Discriminator(nn.Module):
    def __init__(self, img_area=None):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_area, 512),                   ## 输入特征数为784，输出为512
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(512, 256),                        ## 输入特征数为512，输出为256
            nn.LeakyReLU(0.2, inplace=True),            ## 进行非线性映射
            nn.Linear(256, 1),                          ## 输入特征数为256，输出为1
            nn.Sigmoid(),                               ## sigmoid是一个激活函数，二分类问题中可将实数映射到[0, 1],作为概率值, 多分类用softmax函数
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)            ## 鉴别器输入是一个被view展开的(784)的一维图像:(64, 784)
        validity = self.model(img_flat)                 ## 通过鉴别器网络
        return validity                                 ## 鉴别器返回的是一个[0, 1]间的概率
