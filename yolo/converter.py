import torch
import torch.nn as nn
from utils import globalvar as G


__all__ = ['YoloAnchorLayer', 'Yolo2BBox']


class YoloAnchorLayer(nn.Module):
	"""Apply anchors to the output"""
	def __init__(self):
		"""Apply anchors to the output"""
		super(YoloAnchorLayer, self).__init__()
		self.anchors = G.get('anchors')


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""YOLO Anchor Layer
		* Apply anchors to the output
		* Apply sigmoid to the objectness score
		* Apply sigmoid to the class score

		Args:
			X (torch.Tensor): (#, B*(5+num_classes) [num_filter], S, S)

		Returns:
			torch.Tensor: (#, S, S, B*(5+num_classes) [num_filter])
		"""
		self.anchors = self.anchors.to(X.device)
		S = G.get('S')
		B = G.get('B')
		SR = X.shape[2]
		scale_idx = G.get('scale').index(SR / S)
		num_classes = G.get('num_classes')

		# reshape from conv2d shape to (batch_size, S, S, filter)
		X = X.permute(0, 2, 3, 1)
		shape = X.shape

		# reshape to (batch_size, S, S, B, 5 + num_classes) for further processing
		X = X.reshape(-1, SR, SR, B, 5 + num_classes)

		XC = torch.clone(X)

		XC[..., 0:2] = X[..., 0:2].sigmoid()
		XC[..., 4:(5 + num_classes)] = X[..., 4:(5 + num_classes)].sigmoid()
		XC[..., 2] = X[..., 2].exp() * self.anchors[scale_idx][:, 0]
		XC[..., 3] = X[..., 3].exp() * self.anchors[scale_idx][:, 1]
		
		# reshape back
		XC = XC.reshape(shape)
		return XC


class Yolo2BBoxSingle(nn.Module):
	"""convert yolo result from (S, S, (5+?)*B) or (#, S, S, (5+?)*B) to normal bounding box result with size (S*S*B, (5+?)) or (#, S*S*B, (5+?))"""
	def __init__(self):
		"""convert yolo result from (S, S, (5+?)*B) or (#, S, S, (5+?)*B) to normal bounding box result with size (S*S*B, (5+?)) or (#, S*S*B, (5+?))
		
		Note that '5+?' means that the last dimension can be any length greater than or equal to 5, as this trick can be applied to save memory."""
		super(Yolo2BBoxSingle, self).__init__()


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""forward

		Args:
			X (torch.Tensor): yolo result (S, S, (5+?)*B) or (#, S, S, (5+?)*B)

		Returns:
			torch.Tensor: bounding box result (S*S*B, (5+?)) or (#, S*S*B, (5+?)) ((5+?): x1, y1, x2, y2, objectness, class_prob)
		"""
		with torch.no_grad():
			device = X.device
			S = X.shape[1]
			B = G.get('B')

			# arrange cell xidx, yidx
			# [S, S]
			cell_xidx = (torch.arange(S * S) % S).reshape(S, S)
			cell_yidx = (torch.div(torch.arange(S * S), S, rounding_mode='floor')).reshape(S, S)
			# transform to [S, S, B]
			cell_xidx.unsqueeze_(-1)
			cell_yidx.unsqueeze_(-1)
			cell_xidx.expand(S, S, B)
			cell_yidx.expand(S, S, B)
			# move to device
			cell_xidx = cell_xidx.to(device)
			cell_yidx = cell_yidx.to(device)

			single = False
			if len(X.shape) == 3:
				X.unsqueeze_(0)
				single = True

			N = X.shape[0]
			X = X.reshape(N, S, S, B, -1)
			x = (X[..., 0] + cell_xidx) / S
			y = (X[..., 1] + cell_yidx) / S

			x1 = torch.max(x - X[..., 2] / 2.0, torch.zeros_like(x))
			y1 = torch.max(y - X[..., 3] / 2.0, torch.zeros_like(y))
			x2 = torch.min(x + X[..., 2] / 2.0, torch.ones_like(x))
			y2 = torch.min(y + X[..., 3] / 2.0, torch.ones_like(y))

			XC = X.clone()

			XC[..., 0] = x1
			XC[..., 1] = y1
			XC[..., 2] = x2
			XC[..., 3] = y2

			XC = XC.reshape(N, S * S * B, -1)

			if single:
				XC = XC[0]
			
			return XC


class Yolo2BBox(nn.Module):
	"""convert multi-head YOLO result to bbox. accept either single batch or multi batch."""
	def __init__(self):
		"""convert multi-head YOLO result to bbox. accept either single batch or multi batch."""
		super(Yolo2BBox, self).__init__()
		self.converter = Yolo2BBoxSingle()


	def forward(self, X: list[torch.Tensor]) -> torch.Tensor:
		ret = []
		for XS in X:
			ret.append(self.converter(XS))
		return torch.cat(ret, dim=-2)
