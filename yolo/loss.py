import torch
import torch.nn as nn

from utils import globalvar as G
from yolo.converter import Yolo2BBoxSingle


class YoloLoss(nn.Module):
	"""Yolo loss."""
	def __init__(self, 
		lambda_coord: float=1.0, 
		lambda_noobj: float=1.0, 
		lambda_obj: float=1.0, 
		lambda_class: float=1.0, 
		lambda_prior: float=0.0, 
		IoU_thres: float=0.7, 
		epoch_prior: int=0, 
		scale_coord: bool = True, 
		no_obj_v3: bool = True, 
		eps: float = 1e-6):
		"""Yolo loss.

			Written according to https://github.com/AlexeyAB/darknet/issues/821

			Loss = 
				# coordinate loss for responsible bbox
				# prior box loss (used to learn shape of prior boxes)
				# class loss for all bboxes with obj (using only one ground truth)
				# objectness loss for bbox with best IoU less than IoU threshold
				# objectness loss for responsible bbox

			Args:
				yhat (torch.Tensor): yhat, [#, S, S, (5+num_classes)*B], where B is the number of bounding boxes.
				y (torch.Tensor): y, [#, S, S, (5+num_classes)*B], where B is the number of bounding boxes.

			Returns:
				torch.Tensor: loss [#]

		Args:
			lambda_coord (float): lambda for coordinates. Defaults to 1.0.
			lambda_noobj (float): lambda for no_obj, used for objectness. Defaults to 1.0.
			lambda_obj (float): lambda for obj, used for objectness. Defaults to 5.0.
			lambda_class (float): lambda for classes. Defaults to 1.0.
			lambda_prior (float): lambda for prior boxes. Defaults to 0.01.
			IoU_thres (float): IoU threshold while determining no_obj. Defaults to 0.5.
			epoch_prior (int): epoch for learning prior boxes. Defaults to 20.
			scale_coord (bool, optional): whether to scale coordinates (time (2 - w * h)). Defaults to True.
			no_obj_v3 (bool, optional): v3 version of no_obj.
			eps (float, optional): epsilon.
		"""
		super(YoloLoss, self).__init__()
		self.mseloss = nn.MSELoss(reduction='none')
		self.converter = Yolo2BBoxSingle()
		self.lambda_coord = lambda_coord
		self.lambda_noobj = lambda_noobj
		self.lambda_obj = lambda_obj
		self.lambda_class = lambda_class
		self.lambda_prior = lambda_prior
		self.IoU_thres = IoU_thres
		self.epoch_prior = epoch_prior
		self.scale_coord = scale_coord
		self.no_obj_v3 = no_obj_v3
		self.eps = eps


	def cfg_yolov2(self):
		"""use YOLO v2 config"""
		self.lambda_coord = 1.0
		self.lambda_noobj = 1.0
		self.lambda_obj = 5.0
		self.lambda_class = 1.0
		self.lambda_prior = 0.01
		self.IoU_thres = 0.6
		self.epoch_prior = 20
		self.scale_coord = False
		self.no_obj_v3 = False
		self.eps = 1e-6


	def cfg_yolov3(self):
		"""use YOLO v3 config"""
		self.lambda_coord = 1.0
		self.lambda_noobj = 1.0
		self.lambda_obj = 0.0 # YOLO v3 ignore obj loss for IoU higher than thres
		self.lambda_class = 1.0
		self.lambda_prior = 0.0 # YOLO v3 doesn't calculate prior loss
		self.IoU_thres = 0.7
		self.epoch_prior = 0
		self.scale_coord = True
		self.no_obj_v3 = True
		self.eps = 1e-6


	def forward(self, yhat: torch.Tensor, y: torch.Tensor, epoch: int) -> list[torch.Tensor]:
		"""Calculate yolo loss.

		Args:
			yhat (torch.Tensor): yhat, [#, S, S, (num_classes+5)*B], where B is the number of bounding boxes.
			y (torch.Tensor): y, [#, S, S, (num_classes+5)*B], where B is the number of bounding boxes.
			epoch (int): epoch.

		Returns:
			list[torch.Tensor]: [#] coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss
		"""
		S = G.get('S')
		B = G.get('B')
		N = yhat.shape[0]
		SR = y.shape[1]
		num_classes = G.get('num_classes')
		scale_idx = G.get('scale').index(SR / S)
		S = SR

		# [N, S, S, B*(5+num_classes)] => [N, S, S, B, 5+num_classes]
		yhat = yhat.reshape(N, S, S, B, 5 + num_classes)
		y = y.reshape(N, S, S, B, 5 + num_classes)

		with torch.no_grad():
			# convert data into bbox format for IoU calculation
			# half() is also used to save memory
			#    [N, S, S, B, 5+num_classes]
			# => [N, S, S, B, 5]
			# => [N, S*S*B, 5]
			# => [N, S, S, B, 5]
			# YHAT:
			# => [N, S, S, B, 1, 5]
			# => [N, S, S, B (YHat), B (Y), 5]
			# Y:
			# => [N, S, S, 1, B, 5]
			# => [N, S, S, B (YHat), B (Y), 5]
			yhat_bbox = self.converter(yhat[..., 0:5].half()) \
							.reshape(N, S, S, B, 5) \
							.unsqueeze(4) \
							.expand(N, S, S, B, B, 5)
			y_bbox = self.converter(y[..., 0:5].half()) \
							.reshape(N, S, S, B, 5) \
							.unsqueeze(3) \
							.expand(N, S, S, B, B, 5)

			# calculate IoU
			def __intersection__():
				"""internal method to calculate intersection. used to save memory"""
				# [N, S, S, B (YHat), B (Y)]
				wi = torch.min(yhat_bbox[..., 2], y_bbox[..., 2]) - torch.max(yhat_bbox[..., 0], y_bbox[..., 0])
				wi = torch.max(wi, torch.zeros_like(wi))
				hi = torch.min(yhat_bbox[..., 3], y_bbox[..., 3]) - torch.max(yhat_bbox[..., 1], y_bbox[..., 1])
				hi = torch.max(hi, torch.zeros_like(hi))
				return wi * hi

			# [N, S, S, B (YHat), B (Y)]
			intersection = __intersection__()
			union = (yhat_bbox[..., 2] - yhat_bbox[..., 0]) * (yhat_bbox[..., 3] - yhat_bbox[..., 1]) + \
					(y_bbox[..., 2] - y_bbox[..., 0]) * (y_bbox[..., 3] - y_bbox[..., 1]) - intersection
			IoU = intersection / (union + self.eps)

			# [N, S, S, B (YHat), B (Y)] => [N, S, S, B (YHat)]
			MaxIoU = IoU.max(dim=4, keepdim=False)[0]
			# filter out MaxIoU < IoU_thres
			# [N, S, S, B] (boolean index array)
			no_obj_iou = MaxIoU < self.IoU_thres

			# [N, S, S, B (YHat), B (Y)] => [N, S, S, B (Y), B (YHat)]
			IoU = torch.permute(IoU, [0, 1, 2, 4, 3])
			# [N, S, S, B (Y), 1]
			_, idx = IoU.max(dim=4, keepdim=True)

		# width and height (reversed tw and th)
		# [N, S, S, B, 2]
		anchors = G.get('anchors').to(yhat.device)
		wh_hat = torch.log((yhat[:, :, :, :, 2:4] / anchors[scale_idx]) + self.eps)
		wh_true = torch.log((y[:, :, :, :, 2:4] / anchors[scale_idx]) + self.eps)

		# pick responsible data
		# ground truth (y) remain the same
		# detection (yhat) will be reorganized
		# some notes about gather:
		# for dim=3 and index
		# output[N][S][S][B][i] = input[N][S][S][index[N][S][S][B][i]][i]
		yhat_res = yhat.gather(dim=3, index=idx.expand(N, S, S, B, 5 + num_classes))
		wh_hat_res = wh_hat.gather(dim=3, index=idx.expand(N, S, S, B, 2))

		with torch.no_grad():
			# [#, S, S, B]
			have_obj = y[..., 4] > 0

		# calculate loss
		# 1. coordinate loss
		# x and y
		xy_hat = yhat_res[:, :, :, :, 0:2]
		xy_y = y[:, :, :, :, 0:2]
		# calculate loss
		coord_loss = (self.mseloss(xy_hat, xy_y) + self.mseloss(wh_hat_res, wh_true)).sum(dim=4) \
			* have_obj * self.lambda_coord
		if self.scale_coord:
			coord_loss = coord_loss * (2 - y[:, :, :, :, 2] * y[:, :, :, :, 3])
		coord_loss = coord_loss.sum(dim=(1, 2, 3))
		# 2. class loss
		class_loss = self.mseloss(yhat_res[:, :, :, :, 5:], y[:, :, :, :, 5:]).sum(dim=(4)) \
			* have_obj * self.lambda_class
		class_loss = class_loss.sum(dim=(1, 2, 3))
		# 3. no_obj loss
		if self.no_obj_v3:
			no_obj_t = torch.zeros_like(y[:, :, :, :, 4])
		else:
			no_obj_t = y[:, :, :, :, 4]
		no_obj_t = no_obj_t.to(yhat.device)
		no_obj_loss = self.mseloss(yhat[:, :, :, :, 4], no_obj_t) \
			* no_obj_iou * self.lambda_noobj
		no_obj_loss = no_obj_loss.sum(dim=(1, 2, 3))
		# 4. obj loss
		obj_loss = self.mseloss(yhat_res[:, :, :, :, 4], y[:, :, :, :, 4]) \
			* have_obj * self.lambda_obj
		obj_loss = obj_loss.sum(dim=(1, 2, 3))
		# 5. prior loss
		if epoch < self.epoch_prior:
			prior_loss = self.mseloss(yhat[:, :, :, :, 2:4], anchors[scale_idx]).sum(dim=(3, 4)) * self.lambda_prior
			prior_loss = prior_loss.sum(dim=(1, 2))
		else: prior_loss = torch.Tensor([0]).to(yhat.device)

		return coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss
