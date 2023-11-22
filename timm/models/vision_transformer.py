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
import torch
import torch.nn as nn
import cv2
import numpy as np
import skimage
from functools import partial
import time

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_
from .resnet import resnet26d, resnet50d
from .registry import register_model


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # patch models
    "vit_small_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth",
    ),
    "vit_base_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch16_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_base_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_large_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch16_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_large_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_huge_patch16_224": _cfg(),
    "vit_huge_patch32_384": _cfg(input_size=(3, 384, 384)),
    # hybrid models
    "vit_small_resnet26d_224": _cfg(),
    "vit_small_resnet50d_s3_224": _cfg(),
    "vit_base_resnet26d_224": _cfg(),
    "vit_base_resnet50d_224": _cfg(),
}


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding. Finds frequency representation & postion embedding. Does linear projection layer"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class SegmentEmbed(nn.Module):
    """Image to Super-pixel patch embeddings. Converted to frequency domain. Merge number of segments for batched execution. Get positional embeddings
    TODO: Get PCA basis vectors from DataLoader.
    """

    def __init__(
        self,
        segmentation="felz",
        grayscale=True,
        n_points=64,
        num_tokens=196,
        embed_dim=768,
    ):
        super().__init__()
        self.segmentation = (
            segmentation  # segmentation method to use from skimage.semgentation
        )
        self.grayscale = grayscale  # if FT should be taken with grayscale image, or RGB
        self.n_points = n_points  # N-point in FFT
        self.num_tokens = num_tokens  # number of segments to create from image

        self.proj_freq = nn.Linear(self.n_points * self.n_points * 2, embed_dim)
        self.proj_pos = nn.Linear(5, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # Change channels to be last dimension
        print(f"Batch shape is: {x.shape}")
        x = x.cpu().numpy()

        # Iterate over each image in batch and get segmentation mask
        save_mask = np.zeros((B, H, W))
        for i, img in enumerate(x):
            cp_img = np.squeeze(img)

            # TODO: make all parameters set by user
            if self.segmentation == "felz":
                segmentation_mask = skimage.segmentation.felzenszwalb(
                    cp_img, scale=100, sigma=0.5, min_size=50
                ) 
                save_mask[i, :, :] = segmentation_mask

            elif self.segmentation == "slic":
                segmentation_mask = skimage.segmentation.slic(cp_img, n_segments=196, start_label=0)
                save_mask[i, :, :] = segmentation_mask
            else:
                raise ValueError(f"segmentation was set to an invalid method")

        # Globally consistent number of tokens in image
        if self.num_tokens > 0:
            print("Using global consistency for number of tokens")
            num_tokens = self.num_tokens
        # Batch-consistent number of tokens in image
        else:
            print("Using batch consistency for number of tokens")
            # Flatten save_mask. Then find max segment value for each image. Take minimum across all images. Add 1 because of zero-based indexing
            num_tokens = int(np.min(np.max(save_mask.reshape(save_mask.shape[0], -1), axis=1)) + 1)  

        print(f"Number of tokens that will be enforced is: {num_tokens}")
        #### Merge segments to match consistency for batched execution ####
        for i, img in enumerate(x):
            seg_mask = save_mask[i]

            num_segs = np.max(np.unique(seg_mask)) + 1
            print(f"Image: {i} | Segments: {num_segs}")
            assert (
                num_segs >= num_tokens
            ), f"Number of segments in image ({num_segs}) is less than the number of tokens required ({num_tokens}). Please lower the number of tokens or increase segmentation granularity."

            while num_segs > num_tokens:
                # Find smallest segment
                unique_values, counts = np.unique(seg_mask.flatten(), return_counts=True)
                smallest_seg = unique_values[np.argmin(counts)]

                # Find neighbors. Only check for smallest_seg
                vs_right = np.vstack([seg_mask[:,:-1].ravel(), seg_mask[:,1:].ravel()])
                vs_below = np.vstack([seg_mask[:-1,:].ravel(), seg_mask[1:,:].ravel()])
                bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

                include_mask = np.any(bneighbors == smallest_seg, axis=0) # Create a boolean mask indicating columns that include the specific value
                filtered_bneighbors = bneighbors[:, include_mask] # Extract the columns that include the specific value

                # Get unique IDs and remove smallest segment ID
                filtered_bneighbors = np.unique(filtered_bneighbors.flatten())
                filtered_bneighbors = filtered_bneighbors[filtered_bneighbors != smallest_seg]

                # Pick a segment and merge
                merge_seg = np.random.choice(filtered_bneighbors) # randomly pick a segment
                seg_mask[seg_mask == smallest_seg] = merge_seg # merge them

                # Relabel mask
                seg_mask = (skimage.segmentation.relabel_sequential(seg_mask.astype(int)))[0]

                # Update termination condition
                num_segs = np.max(np.unique(seg_mask.flatten())) + 1

        # Create save tensors based on num_tokens
        seg_out = torch.zeros((B, num_tokens, self.n_points * self.n_points * 2))
        pos_out = torch.zeros((B, num_tokens, 5))  # (height, width), (location x,y), area all from [0, 1] -- NOTE: Assuming static positional embeddings for now

        # For each segment in each image
        for j, img in enumerate(x):
            print(j)
            unique_integers = range(int(np.max(save_mask[j])))
            for i, unique_int in enumerate(unique_integers):
                    # TODO: experiment with taking FT of each channel differently vs. only taking grayscale) -- add argument to specify
                    # Get each segment and take FT. Unroll and save
                    binary_mask = (save_mask[j] == unique_int).astype(np.uint8)
                    segmented_img = img * np.expand_dims(binary_mask, axis=-1)

                    # Convert to grayscale if specified and image is RGB
                    if self.grayscale:
                        cp_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

                    # Take FT
                    fourier_transform = np.fft.fft2(cp_img, s=(self.n_points, self.n_points)) #TODO: Change to rfft2 and reduce dims to optimize (as input is always real)
                    # fourier_transform_shifted = np.fft.fftshift(fourier_transform)

                    # Extract magnitude and phase information
                    magnitude = np.abs(fourier_transform)
                    phase = np.angle(fourier_transform)

                    # Save FT info
                    to_save = np.stack((magnitude, phase)).flatten(order="F")                    
                    assert to_save.shape[0] == seg_out.shape[2]
                    seg_out[j, i, :] = torch.from_numpy(
                        to_save
                    ).to('cuda')  # TODO: make this device (how does this work with DataParallel)

                    #### Get position embedding info (static) TODO: Verify implementation ####
                    # Area
                    area = np.sum(binary_mask) / binary_mask.size

                    # Center (Average)
                    center_x = np.average(np.where(binary_mask)[1]) / binary_mask.shape[1]
                    center_y = np.average(np.where(binary_mask)[0]) / binary_mask.shape[0]

                    # Width/Height
                    print(np.where(binary_mask).shape)
                    width = (np.max(np.where(binary_mask)[1]) - np.min(np.where(binary_mask)[1])) / binary_mask.shape[1]
                    height = (np.max(np.where(binary_mask)[0]) - np.min(np.where(binary_mask)[0])) / binary_mask.shape[0]

                    # print(area, area.shape)
                    # print(center_x, center_x.shape)
                    # print(center_y, center_y.shape)
                    # print(width, width.shape)
                    # print(height, height.shape)

                    # Store in array and convert to tensor on device
                    pos_save = torch.from_numpy(np.array([area, center_x, center_y, width, height])).to('cuda') #TODO: make this device (how does this work with DataParallel)
                    pos_out[j, i, :] = pos_save              

        # Add a linear proj layer
        out = self.proj_freq(seg_out) + self.proj_pos(pos_out)

        return out


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
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
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[
                    -1
                ]
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


class SegVisionTransformer(nn.Module):
    """Segmented Vision Transformer"""

    def __init__(
        self,
        segmentation="felz",
        grayscale=True,
        n_points=64,
        num_tokens=196,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        self.patch_embed = SegmentEmbed(
            segmentation=segmentation,
            grayscale=grayscale,
            n_points=n_points,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


# TODO: Check this is correct
@register_model
def segvit_tiny(pretrained=False, **kwargs):
    model = SegVisionTransformer(
        embed_dim=192, num_heads=3, depth=12, mlp_ratio=3, **kwargs
    )
    model.default_cfg = _cfg()

    return model


@register_model
def segvit_small(pretrained=False, **kwargs):
    model = SegVisionTransformer(
        embed_dim=384, num_heads=6, depth=12, mlp_ratio=3, **kwargs
    )
    model.default_cfg = _cfg()

    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault("qk_scale", 768**-0.5)
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.0, **kwargs
    )
    model.default_cfg = default_cfgs["vit_small_patch16_224"]
    if pretrained:
        load_pretrained(
            model,
            num_classes=model.num_classes,
            in_chans=kwargs.get("in_chans", 3),
            filter_fn=_conv_filter,
        )
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_base_patch16_224"]
    if pretrained:
        load_pretrained(
            model,
            num_classes=model.num_classes,
            in_chans=kwargs.get("in_chans", 3),
            filter_fn=_conv_filter,
        )
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_base_patch16_384"]
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get("in_chans", 3)
        )
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_base_patch32_384"]
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get("in_chans", 3)
        )
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_large_patch16_224"]
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get("in_chans", 3)
        )
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_large_patch16_384"]
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get("in_chans", 3)
        )
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_large_patch32_384"]
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get("in_chans", 3)
        )
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs
    )
    model.default_cfg = default_cfgs["vit_huge_patch16_224"]
    return model


@register_model
def vit_huge_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_huge_patch32_384"]
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get(
        "pretrained_backbone", True
    )  # default to True for now, for testing
    backbone = resnet26d(
        pretrained=pretrained_backbone, features_only=True, out_indices=[4]
    )
    model = VisionTransformer(
        img_size=224,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        hybrid_backbone=backbone,
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_small_resnet26d_224"]
    return model


@register_model
def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get(
        "pretrained_backbone", True
    )  # default to True for now, for testing
    backbone = resnet50d(
        pretrained=pretrained_backbone, features_only=True, out_indices=[3]
    )
    model = VisionTransformer(
        img_size=224,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        hybrid_backbone=backbone,
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_small_resnet50d_s3_224"]
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get(
        "pretrained_backbone", True
    )  # default to True for now, for testing
    backbone = resnet26d(
        pretrained=pretrained_backbone, features_only=True, out_indices=[4]
    )
    model = VisionTransformer(
        img_size=224,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        hybrid_backbone=backbone,
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_base_resnet26d_224"]
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get(
        "pretrained_backbone", True
    )  # default to True for now, for testing
    backbone = resnet50d(
        pretrained=pretrained_backbone, features_only=True, out_indices=[4]
    )
    model = VisionTransformer(
        img_size=224,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        hybrid_backbone=backbone,
        **kwargs,
    )
    model.default_cfg = default_cfgs["vit_base_resnet50d_224"]
    return model
