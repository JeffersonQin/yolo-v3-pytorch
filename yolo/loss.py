import torch
import torch.nn as nn

from utils import globalvar as G
from yolo.converter import Yolo2BBox


class YoloLoss(nn.Module):
	"""Yolo loss."""
	def __init__(self, 
		lambda_coord: float, 
		lambda_noobj: float, 
		lambda_obj: float, 
		lambda_class: float, 
		lambda_prior: float, 
		IoU_thres: float, 
		epoch_prior: int, 
		scale_coord: bool = True):
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
		"""
		super(YoloLoss, self).__init__()
		self.lambda_coord = lambda_coord
		self.lambda_noobj = lambda_noobj
		self.lambda_obj = lambda_obj
		self.lambda_class = lambda_class
		self.lambda_prior = lambda_prior
		self.IoU_thres = IoU_thres
		self.epoch_prior = epoch_prior
		self.scale_coord = scale_coord


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
		num_classes = G.get('num_classes')
		SR = y.shape[1]
		scale_idx = G.get('scale').index(SR / S)
		S = SR

		yhat = yhat.reshape(-1, S, S, B, 5 + num_classes)
		y = y.reshape(-1, S, S, B, 5 + num_classes)

		N = yhat.shape[0]

		def internal_function(yhat, y):
			"""used to save memory."""
			with torch.no_grad():
				# convert data into bbox format for IoU calculation
				converter = Yolo2BBox()

				M = yhat.shape[0]

				# half() is also used to save memory
				# [#, S*S*B, 5+num_classes] => [#, S*S*B, 1, 5] => [#, S*S*B, S*S*B, 5]
				yhat_bbox = converter(yhat).half()[..., 0:5].unsqueeze(2).expand(M, S * S * B, S * S * B, 5)
				# [#, S*S*B, 5+num_classes] => [#, 1, S*S*B, 5] => [#, S*S*B, S*S*B, 5]
				y_bbox = converter(y).half()[..., 0:5].unsqueeze(1).expand(M, S * S * B, S * S * B, 5)

				def internal_get_intersection():
					"""used to save memory."""
					# calculate IoU
					# [#, S*S*B, S*S*B]
					wi = torch.min(yhat_bbox[..., 2], y_bbox[..., 2]) - torch.max(yhat_bbox[..., 0], y_bbox[..., 0])
					wi = torch.max(wi, torch.zeros_like(wi))
					hi = torch.min(yhat_bbox[..., 3], y_bbox[..., 3]) - torch.max(yhat_bbox[..., 1], y_bbox[..., 1])
					hi = torch.max(hi, torch.zeros_like(hi))

					# [#, S*S*B (YHat), S*S*B (Y)]
					return wi * hi

				intersection = internal_get_intersection()
				union = (yhat_bbox[..., 2] - yhat_bbox[..., 0]) * (yhat_bbox[..., 3] - yhat_bbox[..., 1]) + \
					(y_bbox[..., 2] - y_bbox[..., 0]) * (y_bbox[..., 3] - y_bbox[..., 1]) - intersection
				IoU = intersection / (union + 1e-12)

				# [#, S*S*B] => [#, S, S, B]
				MaxIoU = IoU.max(dim=2, keepdim=False)[0].reshape(-1, S, S, B)
				# filter out MaxIoU < IoU_thres
				no_obj_iou = MaxIoU < self.IoU_thres

				# [#, S*S*B (YHat), S*S*B (Y)] => [#, S*S*B (YHat), S*S*B (Y)]
				IoU = torch.permute(IoU, [0, 2, 1])

				_, idx = IoU.max(dim=2, keepdim=True)

				return no_obj_iou, idx

		def obtain_by_crop(crop) -> list[torch.Tensor]:
			"""Obtain no_obj_iou by cropping down batch, used to enable large batch training

			Args:
				crop (int): crop count

			Returns:
				list[torch.Tensor]: no_obj_iou and idx
			"""
			no_obj_iou = torch.tensor([], dtype=torch.bool).to(yhat.device)
			idx = torch.tensor([], dtype=torch.int64).to(yhat.device)
			for i in range(crop):
				no_obj_iou_i, idx_i = internal_function(yhat[int(i * N / crop):int((i + 1) * N / crop)], 
														y[int(i * N / crop):int((i + 1) * N / crop)])
				no_obj_iou = torch.cat([no_obj_iou, no_obj_iou_i], dim=0)
				idx = torch.cat([idx, idx_i], dim=0)
			
			return no_obj_iou, idx

		if S == 19:
			crop = 3
		else:
			crop = 1
		
		no_obj_iou, idx = obtain_by_crop(crop)

		# width and height (reversed tw and th)
		anchors = G.get('anchors').to(yhat.device)
		wh_hat = torch.log((yhat[:, :, :, :, 2:4] / anchors[scale_idx]) + 1e-16)
		wh_true = torch.log((y[:, :, :, :, 2:4] / anchors[scale_idx]) + 1e-16)

		if self.scale_coord:
			# coordinate width/height coefficient: (2 - truth.w * truth.h)
			wh_coef = 2 - y[:, :, :, :, 2] * y[:, :, :, :, 3]
		else: wh_coef = 1

		# pick responsible data
		# ground truth (y) remain the same
		# detection (yhat) will be reorganized
		# some notes about gather:
		# for dim=1 and index:
		# output[i][j][k] = input[i][index[i][j][k]][k]
		yhat_res = yhat \
			.reshape(-1, S * S * B, 5 + num_classes) \
			.gather(dim=1, index=idx.expand(N, S * S * B, 5 + num_classes)) \
			.reshape(-1, S, S, B, 5 + num_classes)
		wh_hat_res = wh_hat \
			.reshape(-1, S * S * B, 2) \
			.gather(dim=1, index=idx.expand(N, S * S * B, 2)) \
			.reshape(-1, S, S, B, 2)

		with torch.no_grad():
			# [#, S, S, B]
			have_obj = y[..., 4] > 0

		# calculate loss
		# 1. coordinate loss
		# x and y
		xy_hat = yhat_res[:, :, :, :, 0:2]
		xy_y = y[:, :, :, :, 0:2]
		# calculate loss
		coord_loss = (((xy_hat - xy_y) * wh_coef) ** 2 + ((wh_hat_res - wh_true) * wh_coef) ** 2).sum(dim=4) \
			* have_obj * self.lambda_coord * (2 - y[:, :, :, :, 2] * y[:, :, :, :, 3])
		coord_loss = coord_loss.sum(dim=(1, 2, 3))
		# 2. class loss
		class_loss = ((yhat_res[:, :, :, :, 5:] - y[:, :, :, :, 5:]) ** 2).sum(dim=(4)) \
			* have_obj * self.lambda_class
		class_loss = class_loss.sum(dim=(1, 2, 3))
		# 3. no_obj loss
		no_obj_loss = ((yhat[:, :, :, :, 4] - y[:, :, :, :, 4]) ** 2) \
			* no_obj_iou * self.lambda_noobj
		no_obj_loss = no_obj_loss.sum(dim=(1, 2, 3))
		# 4. obj loss
		obj_loss = (yhat_res[:, :, :, :, 4] - y[:, :, :, :, 4]) ** 2 \
			* have_obj * self.lambda_obj
		obj_loss = obj_loss.sum(dim=(1, 2, 3))
		# 5. prior loss
		if epoch < self.epoch_prior:
			anchors = G.get('anchors').to(yhat.device)
			prior_loss = ((yhat[:, :, :, :, 2:4] - anchors) ** 2).sum(dim=(3, 4)) * self.lambda_prior
			prior_loss = prior_loss.sum(dim=(1, 2))
		else: prior_loss = torch.Tensor([0]).to(yhat.device)

		return coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss
