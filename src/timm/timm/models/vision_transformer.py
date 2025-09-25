""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import time
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import r2_score, accuracy_score
import torch
torch.manual_seed(0)
import torch.nn as nn
from functools import partial, reduce

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_
from .resnet import resnet26d, resnet50d
from .registry import register_model
from .custom_layers import *
# from torchprofile import profile_macs
import os
from PIL import Image


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    mlp_index = 0
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., act_fn_config = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.index = Mlp.mlp_index
        Mlp.mlp_index += 1
        self.bits = int(os.environ['bits'])
        self.topk_percent = float(os.environ['topk_percent'])
        self.threshold = float(os.environ['threshold'])
        self.prev_fp_channels = None
        self.profiling = bool(int(os.environ['profiling']))
        self.oracle = bool(int(os.environ['oracle']))
        self.cache_channel_selection = bool(int(os.environ['cache_channel_selection']))
        self.baseline = bool(int(os.environ['baseline']))
        self.log_dir = str(os.environ['log_dir'])
        self.first_exec = True
        self.max_abs = 0


    def forward(self, x):
        if self.profiling:
            preact = x @ self.fc1.weight.T + self.fc1.bias
            max_abs = math.ceil(torch.max(torch.abs(preact)).item())
            if max_abs > self.max_abs:
                self.max_abs = 1 if max_abs == 0 else 2**math.ceil(math.log2(max_abs))
                print(f"New max abs for layer {self.index}: {self.max_abs}")
                max_abs = self.max_abs
            with open(f"{self.log_dir}/preact_distribution.csv", "a") as f:
                f.write(f"{self.index},{max_abs},16neg,16pos,")
                for i in range(16):
                    ii = 15 - i
                    f.write(f"{torch.sum(torch.where((preact < -ii*max_abs/16) & (preact >= -(ii+1)*max_abs/16), 1, 0)).item()},")

                for i in range(16):
                    f.write(f"{torch.sum(torch.where((preact >= i*max_abs/16) & (preact < (i+1)*max_abs/16), 1, 0)).item()},")
                f.write("\n")

            # weight_max_abs = torch.max(torch.abs(self.fc1.weight), dim=1, keepdim=False).values
            # with open(f"{self.log_dir}/weight_max_distribution.csv", "a") as f:
            #     f.write(f"{self.index},{weight_max_abs.shape[0]},")
            #     for i in range(weight_max_abs.shape[0]):
            #         f.write(f"{weight_max_abs[i].item()},")
            #     f.write("\n")

        if self.baseline:
            outlier_profiling = 0
            if outlier_profiling:
                max_abs_x_dim0 = torch.max(torch.abs(x), dim=0, keepdim=False).values
                mean_max_abs_x_dim0 = torch.mean(max_abs_x_dim0).item()
                outliers_dim0 = torch.sum(torch.where(max_abs_x_dim0 > 2* mean_max_abs_x_dim0, 1, 0)).item()
                max_abs_x_dim1 = torch.max(torch.abs(x), dim=1, keepdim=False).values
                mean_max_abs_x_dim1 = torch.mean(max_abs_x_dim1).item()
                outliers_dim1 = torch.sum(torch.where(max_abs_x_dim1 > 2* mean_max_abs_x_dim1, 1, 0)).item()
                with open(f"{self.log_dir}/input_outliers.csv", "a") as f:
                    f.write(f"{self.index},{max_abs_x_dim0.shape[0]},{outliers_dim0},{mean_max_abs_x_dim0},{max_abs_x_dim1.shape[0]},{outliers_dim1},{mean_max_abs_x_dim1}\n")
                # with open(f"{self.log_dir}/input_max_distribution.csv", "a") as f:
                #     f.write(f"{self.index},{max_abs_x_dim0.shape[0]},{max_abs_x_dim1.shape[0]},")
                #     for i in range(max_abs_x_dim0.shape[0]):
                #         f.write(f"{max_abs_x_dim0[i].item()},")
                #     for i in range(max_abs_x_dim1.shape[0]):
                #         f.write(f"{max_abs_x_dim1[i].item()},")
                #     f.write("\n")

            x = self.fc1(x)
        else:
            topk_rows = int(self.fc1.weight.shape[-1] * self.topk_percent) + 1
            prod_topk = x[:,:,:topk_rows] @ self.fc1.weight[:,:topk_rows].T
            prod = x @ self.fc1.weight.T
            both_neg = torch.where((prod<0) & (prod_topk<0), 1, 0)
            prod_precision = torch.sum(both_neg) / torch.sum(prod<0)
            prod_recall = torch.sum(both_neg) / torch.sum(prod_topk<0)
            prod_topk += self.fc1.bias
            prod += self.fc1.bias

            q_x = quantize(x, self.bits, quantization_type="per_channel")
            q_w = quantize(self.fc1.weight, self.bits, quantization_type="per_channel")
            q_b = quantize(self.fc1.bias, self.bits, quantization_type="per_channel")
            q_x = q_x @ q_w.T + q_b

            if self.oracle:
                x = torch.where(prod<0, q_x, prod)

                experimental = 0
                if experimental:
                    if self.index == 0:
                        x = prod
                    else:
                        x = torch.where(prod<0, q_x, prod)
                        max_abs_x_dim1 = torch.max(torch.abs(x), dim=1, keepdim=True).values
                        mean_max_abs_x_dim1 = torch.mean(max_abs_x_dim1).item()
                        outliers_dim1 = torch.where(max_abs_x_dim1 > mean_max_abs_x_dim1, 1, 0)
                        x = torch.where(outliers_dim1 == 1, prod, x)

                with open(f"{self.log_dir}/oracle_quantization.csv", "a") as f:
                    f.write(f"{self.index},{torch.sum(prod<0).item()},{torch.numel(prod)}\n")

            else:
                threshold = prod_topk.shape[-1] * self.threshold
                num_pos_pred = torch.sum(torch.where(prod_topk>0, 1, 0), dim=(0,1), keepdim=False)
                fp_channels = num_pos_pred > threshold
                num_pos_actual = torch.sum(torch.where(prod>0, 1, 0), dim=(0,1), keepdim=False)
                with open(f"{self.log_dir}/num_pos_channels.csv", "a") as f:
                    f.write(f"{self.index},{len(num_pos_pred.shape)},{len(num_pos_actual.shape)},")
                    if len(num_pos_pred.shape) >= 1:
                        f.write(f"{num_pos_pred.shape[0]},{num_pos_actual.shape[0]},")
                        for i in range(num_pos_pred.shape[0]):
                            f.write(f"{num_pos_pred[i].item()},")
                        for i in range(num_pos_actual.shape[0]):
                            f.write(f"{num_pos_actual[i].item()},")
                    f.write("\n")

                if self.cache_channel_selection:
                    if self.prev_fp_channels is None:
                        self.prev_fp_channels = fp_channels.clone()
                    else:
                        intersection = torch.logical_and(fp_channels, self.prev_fp_channels)
                        channel_selection_stability = torch.sum(intersection).item() / ((torch.sum(self.prev_fp_channels).item() + torch.sum(fp_channels).item()) / 2)
                        with open(f"{self.log_dir}/channel_selection_stability.csv", "a") as f:
                            f.write(f"{self.index},{channel_selection_stability}\n")
                fp_channel_indices = torch.nonzero(fp_channels)

                with open(f"{self.log_dir}/fp_channel_selection.csv", "a") as f:
                    f.write(f"{self.index},{prod_precision},{prod_recall},{torch.sum(fp_channels).item()},{torch.numel(fp_channels)},")
                    # for i in range(fp_channel_indices.shape[0]):
                    #     f.write(f"{fp_channel_indices[i].item()},")
                    f.write("\n")

                x = prod
                fp_channels = fp_channels.expand(x.shape[0], x.shape[1], fp_channels.shape[0])
                x = torch.where(fp_channels, x, q_x)

        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def quantize(input, n_bits, scales = None, quantization_type = "fixed"):
    # n_bits: number of non-sign bits
    # scales: scale for each channel, typically max abs value of the input
    # adapted from https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/fake_quant.py

    if quantization_type == "per_channel":
        scales = input.abs().max(dim=-1, keepdim=True)[0]
    elif quantization_type == "per_tensor":
        scales = input.abs().max()
    elif quantization_type == "per_token":
        input_shape = input.shape
        input.view(-1, input_shape[-1])
        scales = input.abs().max(dim=-1, keepdim=True)[0]

    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max)
    # clip to [-q_max - 1, q_max] when quantizing
    return torch.round(input / scales).clamp(min=-q_max - 1, max=q_max) * scales


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        kqa = KeyQueryAttention(self.num_heads, self.scale, self.qkv, self.attn_drop)
        # macs = profile_macs(kqa, x)
        # with open ("mlp_macs.txt", "a") as f:
        #     f.write(str(macs) + "\n")

        attn = kqa(x)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KeyQueryAttention(nn.Module):
    def __init__(self, num_heads, scale, qkv, attn_drop):
        super().__init__()
        self.num_heads = num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale

        self.qkv = qkv
        self.attn_drop = attn_drop

    def forward(self, x):
        B, N, C = x.shape
        if 1:
            bits = int(os.environ['bits'])
            kqa_topk_prop = float(os.environ['kqa_topk_prop'])
            kqa_threshold = float(os.environ['kqa_threshold'])
            q_x = quantize(x, bits, quantization_type="per_channel")
            q_w = quantize(self.qkv.weight, bits, quantization_type="per_channel")
            q_b = quantize(self.qkv.bias, bits, quantization_type="per_channel")
            q_x = q_x @ q_w.T + q_b
            q_x = quantize(q_x, bits, quantization_type="per_channel")
            q_x = q_x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_q, q_k, q_v = q_x[0], q_x[1], q_x[2]   # make torchscript happy (cannot use tensor as tuple)
            q_attn = (q_q @ q_k.transpose(-2, -1)) * self.scale

            topk_rows = int(self.qkv.weight.shape[-1] * kqa_topk_prop) + 1
            topk_qkv = x[:,:,:topk_rows] @ self.qkv.weight[:,:topk_rows].T + self.qkv.bias
            topk_qkv = topk_qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            topk_q, topk_k, topk_v = topk_qkv[0], topk_qkv[1], topk_qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            topk_rows = int(topk_k.shape[-1] * kqa_topk_prop) + 1
            topk_attn = (topk_q[:,:,:topk_rows] @ topk_k[:,:,:topk_rows].transpose(-2, -1)) * self.scale

        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if 0:
            for i in range(len(attn.shape)):
                print(f"attn dim {i}: {attn.shape[i]}")
                for j in range(attn.shape[i]):
                    tens = torch.index_select(attn, i, torch.tensor([j]).cuda())
                    print(j, torch.sum(torch.where(torch.abs(tens) > kqa_threshold, 1, 0)).item() / torch.numel(tens))
            assert torch.sum(torch.where(torch.abs(attn) > kqa_threshold, 1, 0)) / torch.numel(attn) < 0.1
        if 0:
            attn_shape_prod = reduce(lambda x, y: x * y, attn.shape)
            for i in range(len(attn.shape)):
                non_i_dims = [j for j in range(len(attn.shape)) if j != i]
                sum = torch.sum(torch.where(torch.abs(attn) > kqa_threshold, 1, 0), dim=non_i_dims, keepdim=False) / (attn_shape_prod / attn.shape[i])
                print(f"attn dim {i} shape: {attn.shape[i]}, sum: {sum}")
                sum = torch.sum(torch.where(sum < 0.01, 1, 0)) / attn.shape[i]
                if sum < 0.999999:
                    print(f"attn dim {i}: {attn.shape[i]}, sum: {sum}")


        if 0:
            attn = torch.where(torch.abs(attn) < kqa_threshold, q_attn, attn)
            with open("kqa.csv", "a") as f:
                f.write(f"{torch.sum(torch.where(torch.abs(attn) < kqa_threshold, 0, 1))}, {torch.numel(attn)}\n")


        if 0:
            attn_shape_prod = reduce(lambda x, y: x * y, attn.shape)
            sum = torch.sum(torch.where(topk_attn > kqa_threshold, 1, 0), dim=[0, 2, 3], keepdim=True) / (attn_shape_prod / attn.shape[1])
            # sum = torch.where(sum == torch.max(sum), sum, 0)
            sum_shape = sum.shape
            if torch.sum(sum) == 0:
                sum = torch.zeros_like(sum).to(bool)
            else:
                argmax_sum = torch.argmax(sum.flatten())
                sum = torch.zeros_like(sum.flatten()).to(bool)
                sum[argmax_sum] = True
                sum = sum.reshape(sum_shape)
            with open("kqa_sum.csv", "a") as f:
                f.write(f"{torch.sum(sum).item()}, {torch.numel(sum)}\n")
            # print(sum.shape)
            sum = sum.expand(attn.shape)
            attn = torch.where(sum, attn, q_attn)

        # # print(torch.sum(torch.exp(attn), dim=-1))
        # # print((torch.sum(torch.where(attn <= 0, 1, 0))/torch.numel(attn)).item())

        # q_num_bits_to_calculate = 5
        # k_num_bits_to_calculate = 3
        # max_q_value = 8
        # max_k_value = torch.max(torch.abs(k),dim=-1, keepdim=True).values

        # quantized_dequantized_q = signed_quantization_no_shift(q, q_num_bits_to_calculate, max_q_value)
        # quantized_dequantized_k = signed_quantization_no_shift(k, k_num_bits_to_calculate, max_k_value)
        # quantized_attn = (quantized_dequantized_q @ quantized_dequantized_k.transpose(-2, -1)) * self.scale

        # sum_exp = torch.sum(torch.exp(attn), dim=-1, keepdim=True)

        # sum_exp_attn = torch.where(sum_exp > 200.0, quantized_attn, attn)

        # # sum_exp_large = torch.where(sum_exp > 200.0, 1, 0)
        # # quantized_self.non_pos = torch.where(quantized_attn <= 0.0, 1, 0)
        # # print((torch.sum(sum_exp_large*quantized_self.non_pos)/torch.numel(quantized_self.non_pos)).item())


        # attn = torch.where(quantized_attn <= 0.0, sum_exp_attn, attn)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        return attn


def signed_quantization_no_shift(matrix, num_non_signbits, max_value):
    num_values_in_quantization = 2**num_non_signbits
    max_value_in_quantization = num_values_in_quantization - 1
    return torch.sign(matrix) * torch.min(torch.round((torch.abs(matrix))/max_value * num_values_in_quantization) + 1, torch.tensor(max_value_in_quantization).cuda())

def reverse_signed_quantization_no_shift(matrix, num_non_signbits, max_value):
    num_values_in_quantization = 2**num_non_signbits
    return torch.sign(matrix) * (torch.abs(matrix) - 1) / num_values_in_quantization * max_value
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, act_fn_config=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, act_fn_config=act_fn_config)

    def forward(self, x):

        # macs = profile_macs(self.attn, self.norm1(x))
        # with open ("mlp_macs.txt", "a") as f:
        #     f.write(str(macs) + "\n")
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # macs = profile_macs(self.mlp, x)
        # print("MLP MACs:", macs)
        # with open ("mlp_macs.txt", "a") as f:
        #     f.write(str(macs) + "\n")
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, act_fn_config=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_fn_config=act_fn_config)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        # macs = profile_macs(self.patch_embed, x)
        # with open ("mlp_macs.txt", "a") as f:
        #     f.write(str(macs) + "\n")

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        # macs = profile_macs(self.head, x)
        # with open ("mlp_macs.txt", "a") as f:
        #     f.write(str(macs) + "\n")
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch16_224']
    return model


@register_model
def vit_huge_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch32_384']
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet26d_224']
    return model


@register_model
def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet50d_s3_224']
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet26d_224']
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet50d_224']
    return model
