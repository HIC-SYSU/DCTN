import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

import nibabel
import SimpleITK as sitk
import numpy as np
import skan
from skimage.morphology import skeletonize_3d, binary_opening, binary_closing, ball
from vtkplotter import *


import os
import nrrd
import csv
import nrrd
import pandas as pd
from numpy.linalg import svd

""" 
Extractor
"""
class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))

class DenseBlock(nn.Module):
    def __init__(self, input_channels, num_layers, growth_rate):
            super(DenseBlock, self).__init__()
            self.num_layers = num_layers
            self.k0 = input_channels
            self.k = growth_rate
            self.layers = self.__make_layers()

    def __make_layers(self):
            layer_list = []
            for i in range(self.num_layers):
                layer_list.append(nn.Sequential(
                    BN_Conv2d(self.k0 + i * self.k, 4 * self.k, 1, 1, 0),
                    BN_Conv2d(4 * self.k, self.k, 3, 1, 1)
                ))
            return layer_list

    def forward(self, x):
            feature = self.layers[0](x)
            out = torch.cat((x, feature), 1)
            for i in range(1, len(self.layers)):
                feature = self.layers[i](out)
                out = torch.cat((feature, out), 1)
            return out

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, patch_count=16, in_chans=1, embed_dim=512,):
        super().__init__()
        patch_stride = img_size // patch_count
        patch_padding = (patch_stride * (patch_count - 1) + patch_size - img_size) // 2
        num_patches = patch_count ** 2

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.feature = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_padding)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.feature(x).flatten(2).transpose(1, 2)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=16, bias=False, scale=None, feature_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(dim, dim * 2, bias=bias)
        self.feature = nn.Linear(dim, dim)
        self.feature_drop = nn.Dropout(feature_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.feature(x)
        x = self.feature_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class DDB(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., bias=False, scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads, bias=bias, scale=scale,featurte_drop=drop )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class ICM(nn.Module):
    def init(self):
        super(ICM, self).init()
        selfresnet = ResBlock(n_chans=256)
        self.feature = nn.Sequential(*list(self.resnet.children())[:-2])
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, current_img, prev_imgs, next_imgs):
        current_features = self.feature_extractor(current_img)
        prev_features = [self.feature_extractor(img) for img in prev_imgs]
        next_features = [self.feature_extractor(img) for img in next_imgs]

        current_features = self.conv1x1(current_features)
        current_features = self.relu(current_features)

        prev_corr = self.calculate_correlation(current_features, prev_features)
        next_corr = self.calculate_correlation(current_features, next_features)

        semantic_features = torch.cat((current_features, prev_corr, next_corr), dim=1)
        return semantic_features

    def calculate_correlation(self, current_features, other_features):
        batch_size, channels, height, width = current_features.size()
        correlation_map = torch.zeros(batch_size, 1, height, width).to(current_features.device)

        for features in other_features:
            correlation_map += torch.sum(current_features * features, dim=1, keepdim=True)

        return correlation_map

class PatchPoint(nn.Module):
    def init(self, input_size, patch_count, tanh=True,weights=(1., 1.)):
        super(PatchPoint, self).init()
        self.input_size = input_size
        self.patch_count = patch_count
        self.generate_NewPoint()
        self.tanh = tanh
        self.weights = weights

    def generate_NewPoint(self):
        NewPoint = []
        patch_stride = 1. / self.patch_count
        for i in range(self.patch_count):
            for j in range(self.patch_count):
                y = (0.5+i)*patch_stride
                x = (0.5+j)*patch_stride
                NewPoint.append([x, y])
        NewPoint = torch.as_tensor(NewPoint)
        self.register_buffer("NewPoint", NewPoint)

    def generate_offsets(self):
        return (self.NewPoints - self.NewPoint) * self.input_size

    def resimple(self, rel_codes):
        NewPoints=self.NewPoint
        pixel = 1. / self.patch_count
        wx, wy = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel if self.tanh else rel_codes[:, :, 0] * pixel / wx
        dy = F.tanh(rel_codes[:, :, 1] / wy) * pixel if self.tanh else rel_codes[:, :, 1] * pixel / wy

        pred_NewPoint = torch.zeros_like(rel_codes)

        ref_x = NewPoints[:, 0].unsqueeze(0)
        ref_y = NewPoints[:, 1].unsqueeze(0)

        pred_NewPoint[:, :, 0] = dx + ref_x
        pred_NewPoint[:, :, 1] = dy + ref_y
        pred_NewPoint = pred_NewPoint.clamp_(min=0., max=1.)

        return pred_NewPoint

    def forward(self, patchs, model_offset=None):
        self.NewPoint = self.resimple(patchs)
        return self.NewPoint

class Extractor(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, num_classes=2, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], patch_embeds=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4 = patch_embeds

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches + 1, embed_dims[3]))

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        CS = 0
        self.block1 = nn.ModuleList([DDB(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], bias=bias, scale=scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[CS + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        CS += depths[0]
        self.block2 = nn.ModuleList([DDB(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], bias=bias, scale=scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[CS + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        CS += depths[1]
        self.block3 = nn.ModuleList([DDB(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], bias=bias, scale=scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[CS + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        CS += depths[2]
        self.block4 = nn.ModuleList([DDB(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], bias=bias, scale=scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[CS + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm = norm_layer(embed_dims[3])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # new
        self.patch_embed2.reset_offset()
        self.patch_embed3.reset_offset()
        self.patch_embed4.reset_offset()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        CS = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[CS + i]

        CS += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[CS + i]

        CS += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[CS + i]

        CS += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[CS + i]

    def forward_features(self, x):
        B = x.shape[0]
        aux_results = []

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        if isinstance(x, tuple):
            aux_results.append(x[1])
            x = x[0]
        x = x + self.pos_embed2
        x = self.pos_dros2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        if isinstance(x, tuple):
            aux_results.append(x[1])
            x = x[0]
        x = x + self.pos_embed3
        x = self.pos_dros3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        if isinstance(x, tuple):
            aux_results.append(x[1])
            x = x[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed4
        x = self.pos_dros4(x)
        for blk in self.block4:
            x = blk(x, H, W)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


""" 
Decoder
"""
class GDBlock(nn.Module):
    def __init__(self, input_channels):
        super(GDBlock, self).__init__()
        self.input_channels = input_channels
        self.out_channels = int(input_channels / 2)

        self.s1_feature = nn.Sequential(
            nn.Conv2d(self.input_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())        
        self.s1 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s1_new = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        
        self.s2_feature = nn.Sequential(
            nn.Conv2d(self.input_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s2_new = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        
        self.s3_feature = nn.Sequential(
            nn.Conv2d(self.input_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s3 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 5, 1, 2),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s3_new = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())

        self.gd_results = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        s1_input = self.s1_feature(x)
        s1 = self.s1(s1_input)
        s1_new = self.s1_new(s1)

        s2_input = self.s2_feature(x) + s1_new
        s2 = self.s2(s2_input)
        s2_new = self.s2_new(s2)

        s3_input = self.s3_feature(x) + s2_new
        s3 = self.s3(s3_input)
        s3_new = self.s3_new(s3)

        out = self.gd_results(torch.cat((s1_new, s2_new, s3_new), 1))

        return out

class Decoder(nn.Module):
    def __init__(self, channel1, channel2):
        super(Decoder, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.input_high_level_feature = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 5, 1, 2),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.beta_1 = nn.Parameter(torch.ones(1))
        self.beta_2 = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()
        self.output_map = nn.Conv2d(self.channel1, 1, 5, 1, 2)

    def forward(self, x, y, higher_level_prediction):
        # /upsample
        up = self.up(y)

        input_high_level_feature = self.input_high_level_feature(higher_level_prediction)
        foreground_feature = x * input_high_level_feature
        background_feature = x * (1 - input_high_level_feature)

        f1 = GDBlock(foreground_feature)
        f2 = GDBlock(background_feature)

        refine_feature1 = up - (self.beta_2 * f1)
        refine_feature1 = self.bn1(refine_feature1)
        refine_feature1 = self.relu1(refine_feature1)

        refine_feature2 = refine_feature1 + (self.beta_1 * f2)
        refine_feature2 = self.bn2(refine_feature2)
        refine_feature2 = self.relu2(refine_feature2)

        output_map = self.output_map(refine_feature2)

        return refine_feature2, output_map

""" 
Guider
"""
# get centerline
def get_centerline(image_path):

    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)

    vtkImage = Volume(image)
    s = vtkImage.isosurface(1).alpha(0.5)

    image = binary_opening(image)
    image=vtk.vtkImageData(data)
    image = vtk.vtkImageGaussianSmooth(image)

    skele = skeletonize_3d(image)
    skeletonNet = skan.Skeleton(skele)

    paths = [skeletonNet.path_coordinates(i) for i in range(skeletonNet.paths.shape[0])]
    #print(paths)

    s = [s]
    for path in paths:
        p = np.array(path)
        if p.shape[0] == 2:
            spl = Spline(p, smooth=1, degree=1)
        else:
            spl = Spline(p, smooth=1)
        s.append(spl)
    # show(s, axes=0)
    return s

# CPR based on 3D slicer
def curveLength(x,y,z):
    diffs = np.sqrt(np.diff(x)**2+np.diff(y)**2+np.diff(z)**2)
    length = diffs.sum()

    return length

def planeFit(points):

    points = points.T # dimension is (N,3)
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T)
    return ctr, svd(M)[0][:,-1]

def readPoints():
    pandas_data = pd.read_csv(r'path', sep=',', skiprows=3)
    numpy_data = np.asarray(pandas_data)
    all_points = np.asarray(numpy_data[:,1:4], 'float')
    return all_points

def computeStraighteningTransform(curvePoints, outputSpacingMm = 0.1):

    transformation_RAS2ijk = np.array([[-0.287, 0 ,0, 73.212],[0, -0.287, 0, 75.644],[0, 0 ,0.287, -80.779],[0 ,0 ,0 ,1]]) # 1。gotten from DICOM header
    transformation_RAS2ijk = np.array([[-0.287, 0, 0, 73.212], [0, -0.287, 0, 75.644], [0, 0, 0.287, -80.779],
                                       [-1, -1, 1, 1]])  # gotten from DICOM header

    # Slice thickness characterizes how sharply focused your image slice is. can be found in metadata of CT
    sliceSizeMm = [0.5,  0.5, 0.5]
    curve_length = curveLength(curvePoints[:,0], curvePoints[:,1], curvePoints[:,2])
    nPoints = curvePoints.shape[0]

    transformSpacingFactor = 5

    resamplingCurveSpacing = outputSpacingMm * transformSpacingFactor

# Z axis (from first curve point to last, this will be the straightened curve long axis)
#transformGridAxisZ = (curveEndPoint-curveStartPoint)/np.linalg.norm(curveEndPoint-curveStartPoint)
    transformGridAxisZ = (curvePoints[-1,:]-curvePoints[0,:])/np.linalg.norm(curvePoints[-1,:]-curvePoints[0,:])


# X axis = average X axis of curve
    sumCurveAxisX_RAS = np.zeros(3)
    for nSlices in range(nPoints):

        curvePointToWorldArray = transformation_RAS2ijk

        curveAxisX_RAS = curvePointToWorldArray[0:3, 0]
        sumCurveAxisX_RAS += curvePoints[nSlices,:]

        meanCurveAxisX_RAS = sumCurveAxisX_RAS/np.linalg.norm(sumCurveAxisX_RAS)
        transformGridAxisX = meanCurveAxisX_RAS

# Y axis = average Y axis of curve
    transformGridAxisY = np.cross(transformGridAxisZ, transformGridAxisX)
    transformGridAxisY = transformGridAxisY/np.linalg.norm(transformGridAxisY)

# Make sure that X axis is orthogonal to Y and Z
    transformGridAxisX = np.cross(transformGridAxisY, transformGridAxisZ)
    transformGridAxisX = transformGridAxisX/np.linalg.norm(transformGridAxisX)


# Origin (makes the grid centered at the curve)
    planeCenter, norm = planeFit(curvePoints)

    transformGridOrigin = planeCenter
    transformGridOrigin -= transformGridAxisX * sliceSizeMm[0]/2.0
    transformGridOrigin -= transformGridAxisY * sliceSizeMm[1]/2.0
    transformGridOrigin -= transformGridAxisZ * curve_length/2.0

    gridDimensions = [2, 2, nPoints]
    gridSpacing = [sliceSizeMm[0], sliceSizeMm[1], resamplingCurveSpacing]
    gridDirectionMatrixArray = np.eye(4)
    gridDirectionMatrixArray[0:3, 0] = transformGridAxisX
    gridDirectionMatrixArray[0:3, 1] = transformGridAxisY
    gridDirectionMatrixArray[0:3, 2] = transformGridAxisZ


    transformDisplacements_RAS = np.zeros((gridDimensions[0],gridDimensions[1],gridDimensions[2]))

    for gridK in range(gridDimensions[2]):

            curvePointToWorldArray = transformation_RAS2ijk

            curveAxisX_RAS = curvePointToWorldArray[0:3, 0]
            curveAxisY_RAS = curvePointToWorldArray[0:3, 1]
            curvePoint_RAS = curvePointToWorldArray[0:3, 3]

    for gridJ in range(gridDimensions[1]):
        for gridI in range(gridDimensions[0]):
            straightenedVolume_RAS = (transformGridOrigin+ gridI*gridSpacing[0]*transformGridAxisX+ gridJ*gridSpacing[1]*transformGridAxisY+ gridK*gridSpacing[2]*transformGridAxisZ)
            inputVolume_RAS = (curvePoint_RAS + (gridI-0.287)*sliceSizeMm[0]*curveAxisX_RAS + (gridJ-0.287)*sliceSizeMm[1]*curveAxisY_RAS)
            transformDisplacements_RAS[gridK][gridJ][gridI] = inputVolume_RAS - straightenedVolume_RAS # TODO ValueError: setting an array element with a sequence.

            return transformDisplacements_RAS

# graph optimal transport
def cost_matrix(x, y):
    x = x.div(torch.norm(x, p=2, dim=0, keepdim=True) )
    y = y.div(torch.norm(y, p=2, dim=0, keepdim=True) )
    cost_distance = 1 - torch.mm(torch.transpose(y, 0, 1), x)
    return cost_distance

def OT(C, n, m, miu, nu, beta=0.5):
	# C is the distance matrix
	sigma = torch.ones(int(m), 1).float().cuda()/m
	T = torch.ones(n, m).cuda()
	C = torch.exp(-C/beta).float()
	for t in range(15):
		T = C * T
		for k in range(1):
			delta = miu / torch.squeeze(torch.matmul(T, sigma))
			sigma = torch.unsqueeze(nu,1) / torch.matmul(torch.transpose(T,0,1), torch.unsqueeze(delta,1))
		T = torch.unsqueeze(delta,1) * T * sigma.transpose(1,0)
	return T.detach()

def W_distance(C, n, m):
    T = OT(C, n, m)
    distance = torch.trace(torch.matmul(C, T.t()))
    return W_distance