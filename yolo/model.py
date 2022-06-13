from typing import Tuple
import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from utils import globalvar as G
from utils.winit import weight_init
from yolo.converter import YoloAnchorLayer


class ConvUnit(nn.Module):
	"""Convolutional Unit, consists of conv2d, batchnorm, leaky_relu"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(ConvUnit, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.bn = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
	
	def forward(self, X: torch.Tensor) -> torch.Tensor:
		X = self.conv(X)
		X = self.bn(X)
		X = self.leaky_relu(X)
		return X


class YoloBlock(nn.Module):
	"""YOLO Conv Block, consists of 2 * [ 1x1-n ConvUnit, 3x3-2n ConvUnit ] + 1x1-n ConvUnit"""
	def __init__(self, in_channels, out_channels):
		super(YoloBlock, self).__init__()
		self.conv1x1_1 = ConvUnit(in_channels, out_channels, 1, 1, 0)
		self.conv3x3_1 = ConvUnit(out_channels, out_channels * 2, 3, 1, 1)
		self.conv1x1_2 = ConvUnit(out_channels * 2, out_channels, 1, 1, 0)
		self.conv3x3_2 = ConvUnit(out_channels, out_channels * 2, 3, 1, 1)
		self.conv1x1_3 = ConvUnit(out_channels * 2, out_channels, 1, 1, 0)
		self.unit = nn.Sequential(self.conv1x1_1, self.conv3x3_1, self.conv1x1_2, self.conv3x3_2, self.conv1x1_3)


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		return self.unit(X)


class ResNetYolo(nn.Module):
	"""YOLO using ResNet backbone"""
	def __init__(self, resnet_backbone: nn.Module):
		super(ResNetYolo, self).__init__()
		self.backbone = create_feature_extractor(resnet_backbone, return_nodes={
			'layer2': 'scale_4',
			'layer3': 'scale_2',
			'layer4': 'scale_1',
		})

		# obtain necessary filter settings
		B = G.get('B')
		num_classes = G.get('num_classes')
		num_features = B * (5 + num_classes)
		with torch.no_grad():
			# dry run to get number of channels
			inpt = torch.randn(1, 3, 224, 224)
			out = self.backbone(inpt)
			scale_4_channels = out['scale_4'].shape[1]
			scale_2_channels = out['scale_2'].shape[1]
			scale_1_channels = out['scale_1'].shape[1]

		self.anchor = YoloAnchorLayer()
		# tail
		self.scale_1_tail = nn.Sequential(ConvUnit(512, 1024, 3, 1, 1), ConvUnit(1024, num_features, 1, 1, 0), self.anchor)
		self.scale_2_tail = nn.Sequential(ConvUnit(256, 512, 3, 1, 1), ConvUnit(512, num_features, 1, 1, 0), self.anchor)
		self.scale_4_tail = nn.Sequential(ConvUnit(128, 256, 3, 1, 1), ConvUnit(256, num_features, 1, 1, 0), self.anchor)
		# head
		self.scale_1_head = YoloBlock(scale_1_channels, 512)
		self.scale_2_head = YoloBlock(scale_2_channels + 256, 256)
		self.scale_4_head = YoloBlock(scale_4_channels + 128, 128)
		# pass through
		self.scale_1_pass_through = nn.Sequential(ConvUnit(512, 256, 1, 1, 0), nn.UpsamplingBilinear2d(scale_factor=2))
		self.scale_2_pass_through = nn.Sequential(ConvUnit(256, 128, 1, 1, 0), nn.UpsamplingBilinear2d(scale_factor=2))


	def winit(self, pretrained=True):
		self.scale_1_tail.apply(weight_init)
		self.scale_2_tail.apply(weight_init)
		self.scale_4_tail.apply(weight_init)
		self.scale_1_head.apply(weight_init)
		self.scale_2_head.apply(weight_init)
		self.scale_4_head.apply(weight_init)
		self.scale_1_pass_through.apply(weight_init)
		self.scale_2_pass_through.apply(weight_init)
		if not pretrained:
			self.backbone.apply(weight_init)


	def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		out = self.backbone(X)
		scale_4 = out['scale_4']
		scale_2 = out['scale_2']
		scale_1 = out['scale_1']

		scale_1_mid = self.scale_1_head(scale_1)
		scale_1_out = self.scale_1_tail(scale_1_mid)

		scale_2_in_pass = self.scale_1_pass_through(scale_1_mid)
		scale_2_in = torch.cat([scale_2, scale_2_in_pass], dim=1)
		scale_2_mid = self.scale_2_head(scale_2_in)
		scale_2_out = self.scale_2_tail(scale_2_mid)

		scale_4_in_pass = self.scale_2_pass_through(scale_2_mid)
		scale_4_in = torch.cat([scale_4, scale_4_in_pass], dim=1)
		scale_4_mid = self.scale_4_head(scale_4_in)
		scale_4_out = self.scale_4_tail(scale_4_mid)

		return scale_1_out, scale_2_out, scale_4_out


class ResNet18YoloDetector(ResNetYolo):
	def __init__(self, pretrain=True):
		super(ResNet18YoloDetector, self).__init__(torchvision.models.resnet18(pretrained=pretrain))


	def forward(self, X):
		return super(ResNet18YoloDetector, self).forward(X)


class ResNet34YoloDetector(ResNetYolo):
	def __init__(self, pretrain=True):
		super(ResNet34YoloDetector, self).__init__(torchvision.models.resnet34(pretrained=pretrain))


	def forward(self, X):
		return super(ResNet34YoloDetector, self).forward(X)


class ResNet50YoloDetector(ResNetYolo):
	def __init__(self, pretrain=True):
		super(ResNet50YoloDetector, self).__init__(torchvision.models.resnet50(pretrained=pretrain))


	def forward(self, X):
		return super(ResNet50YoloDetector, self).forward(X)


class ResNet101YoloDetector(ResNetYolo):
	def __init__(self, pretrain=True):
		super(ResNet101YoloDetector, self).__init__(torchvision.models.resnet101(pretrained=pretrain))


	def forward(self, X):
		return super(ResNet101YoloDetector, self).forward(X)


class ResNet152YoloDetector(ResNetYolo):
	def __init__(self, pretrain=True):
		super(ResNet152YoloDetector, self).__init__(torchvision.models.resnet152(pretrained=pretrain))


	def forward(self, X):
		return super(ResNet152YoloDetector, self).forward(X)