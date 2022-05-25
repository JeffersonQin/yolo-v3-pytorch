import os
import random
from typing import Optional
import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard

from torch.utils.data import DataLoader
from utils import metrics as metrics_utils
from utils.utils import Accumulator, Timer, get_all_gpu, update_lr
from utils import G
from yolo.loss import YoloLoss


def train(net: nn.Module, train_iter: DataLoader, test_iter: DataLoader, num_epochs: int, multi_scale_epoch: int, output_scale_S: int, lr, optimizer: torch.optim.Optimizer, log_id: str, loss=YoloLoss(), num_gpu: int=1, accum_batch_num: int=1, save_dir: str='./model', load_model: Optional[str]=None, load_optim: Optional[str]=None, load_epoch: int=-1, visualize_cnt: int=10):
	"""trainer for yolo v2. 
	Note: weight init is not done in this method, because the architecture
	of yolo v2 is rather complicated with the design of pass through layer

	Args:
		net (nn.Module): module network
		train_iter (DataLoader): training dataset iterator
		test_iter (DataLoader): testing dataset iterator
		num_epochs (int): number of epochs to train
		multi_scale_epoch (int): number of epochs to train with multi scale
		output_scale_S (int): final network scale (S), input size will be 32S * 32S, as the network stride is 32
		lr (float | callable): learning rate or learning rate scheduler function relative to epoch
		optimizer (torch.optim.Optimizer): optimizer
		log_id (str): identifier for logging in tensorboard.
		loss (YoloLoss): loss function
		num_gpu (int, optional): number of gpu to train on, used for parallel training. Defaults to 1.
		accum_batch_num (int, optional): number of batch to accumulate gradient, used to solve OOM problem when using big batch sizes. Defaults to 1.
		save_dir (str, optional): saving directory for model weights. Defaults to './model'.
		load_model (Optional[str], optional): path of model weights to load if exist. Defaults to None.
		load_optim (Optional[str], optional): path of optimizer state_dict to load if exist. Defaults to None.
		load_epoch (int, optional): done epoch count minus one when loading, should be the same with the number in auto-saved file name. Defaults to -1.
		visualize_cnt (int, optional): number of batches to visualize each epoch during training progress. Defaults to 10.
	"""
	os.makedirs(save_dir, exist_ok=True)

	# tensorboard
	writer = tensorboard.SummaryWriter(f'logs/yolo')
	pr_writer = tensorboard.SummaryWriter(f'logs/yolo/pr/{log_id}')

	# set up loading
	if load_model:
		net.load_state_dict(torch.load(load_model))

	# set up devices
	if not torch.cuda.is_available():
		net = net.to(torch.device('cpu'))
		devices = [torch.device('cpu')]
	else:
		if num_gpu > 1:
			net = nn.DataParallel(net, get_all_gpu(num_gpu))
			devices = get_all_gpu(num_gpu)
		else:
			net = net.to(torch.device('cuda'))
			devices = [torch.device('cuda')]

	if load_optim:
		optimizer.load_state_dict(torch.load(load_optim))

	num_batches = len(train_iter)
	num_scales = len(G.get('scale'))

	def plot(batch: int, num_batches: int, visualize_cnt: int) -> int:
		"""judge whether to plot or not for a specific batch

		Args:
			batch (int): batch count (starts from 1)
			num_batches (int): total batch count
			visualize_cnt (int): how many batches to visualize for each epoch

		Returns:
			int: if plot, return plot indicies (1 ~ visualize_cnt), else return 0
		"""
		if num_batches % visualize_cnt == 0:
			if batch % (num_batches // visualize_cnt) == 0:
				return batch // (num_batches // visualize_cnt)
			else:
				return 0
		else:
			if batch % (num_batches // visualize_cnt) == 0:
				if batch // (num_batches // visualize_cnt) == visualize_cnt:
					return 0
				else:
					return batch // (num_batches // visualize_cnt)
			elif batch == num_batches:
				return visualize_cnt
			else:
				return 0


	def log_loss_tensorboard(metrics: list[Accumulator], epoch: int, visualize_cnt: int, plot_indices: int, j: int, train: bool):
		"""log loss to tensorboard

		Args:
			metrics (list[Accumulator]): metrics list to log
			epoch (int): epoch count
			visualize_cnt (int): visualize cnt
			plot_indices (int): plot indices
			j (int): scale index
			train (bool): if train or test
		"""
		if train: prefix = 'train'
		else: prefix = 'test'
		if j == num_scales: suffix = 'all'
		else: suffix = f'scale-{G.get("scale")[j]}'
		writer.add_scalars(f'loss/{log_id}/total-{suffix}', {prefix: metrics[j][5] / metrics[j][6],}, epoch * visualize_cnt + plot_indices)
		writer.add_scalars(f'loss/{log_id}/coord-{suffix}', {prefix: metrics[j][0] / metrics[j][6],}, epoch * visualize_cnt + plot_indices)
		writer.add_scalars(f'loss/{log_id}/class-{suffix}', {prefix: metrics[j][1] / metrics[j][6],}, epoch * visualize_cnt + plot_indices)
		writer.add_scalars(f'loss/{log_id}/no_obj-{suffix}', {prefix: metrics[j][2] / metrics[j][6],}, epoch * visualize_cnt + plot_indices)
		writer.add_scalars(f'loss/{log_id}/obj-{suffix}', {prefix: metrics[j][3] / metrics[j][6],}, epoch * visualize_cnt + plot_indices)
		writer.add_scalars(f'loss/{log_id}/prior-{suffix}', {prefix: metrics[j][4] / metrics[j][6],}, epoch * visualize_cnt + plot_indices)

	# train
	for epoch in range(num_epochs - load_epoch - 1):
		# adjust true epoch number according to pre_load
		epoch = epoch + load_epoch + 1

		# define metrics: coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss, train loss, sample count
		metrics = [Accumulator(7) for _ in range(num_scales + 1)]
		# define timer
		timer = Timer()

		# train
		net.train()

		# set batch accumulator
		accum_cnt = 0
		accum = 0

		# iterate over batches
		timer.start()
		for i, (X, y) in enumerate(train_iter):
			X, y = X.to(devices[0]), [ys.to(devices[0]) for ys in y]
			with torch.autocast(devices[0]):
				yhat = net(X)
				plot_indices = plot(i + 1, num_batches, visualize_cnt)
				total_loss = 0

				for j, (y_single, yhat_single) in enumerate(zip(y, yhat)):
					coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss = loss(yhat_single, y_single, epoch)
					loss_val = coord_loss + class_loss + no_obj_loss + obj_loss + prior_loss

					total_loss = total_loss + loss_val.sum()

					with torch.no_grad():
						metrics[j].add(coord_loss.sum(), class_loss.sum(), no_obj_loss.sum(), obj_loss.sum(), prior_loss.sum(), loss_val.sum(), X.shape[0])
						metrics[num_scales].add(coord_loss.sum(), class_loss.sum(), no_obj_loss.sum(), obj_loss.sum(), prior_loss.sum(), loss_val.sum(), 0)

					# log train loss
					print(f'epoch {epoch} batch {i + 1}/{num_batches} scale {G.get("scale")[j]} loss: {metrics[j][5] / metrics[j][6]}, S: {G.get("S")}, B: {G.get("B")}')
					if plot_indices > 0:
						log_loss_tensorboard(metrics, epoch, visualize_cnt, plot_indices, j, train=True)

				# log total scale loss
				metrics[num_scales].add(0, 0, 0, 0, 0, 0, X.shape[0])
				print(f'epoch {epoch} batch {i + 1}/{num_batches} total loss: {metrics[num_scales][5] / metrics[num_scales][6]}, S: {G.get("S")}, B: {G.get("B")}')
				if plot_indices > 0:
					log_loss_tensorboard(metrics, epoch, visualize_cnt, plot_indices, num_scales, train=True)

			# backward to accumulate gradients
			total_loss.sum().backward()

			# update batch accumulator
			accum += 1
			accum_cnt += X.shape[0]
			# step when accumulator is full
			if accum == accum_batch_num or i == num_batches - 1:
				# update learning rate per epoch and adjust by accumulated batch_size
				if callable(lr):
					update_lr(optimizer, lr(epoch) / accum_cnt)
				else:
					update_lr(optimizer, lr / accum_cnt)
				# step
				optimizer.step()
				# clear
				optimizer.zero_grad()
				accum_cnt = 0
				accum = 0

			# random choose a new image dimension size from
			# [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
			# that is, randomly adjust S between [10, 19]
			if (i + 1) % 10 == 0 and epoch < multi_scale_epoch:
				# make last two batches to adjust S
				if i + 21 >= num_batches:
					G.set('S', output_scale_S)
				else:
					G.set('S', random.randint(10, 19))
			elif epoch >= multi_scale_epoch:
				G.set('S', output_scale_S)

		timer.stop()
		# log train timing
		writer.add_scalars(f'timing/{log_id}', {'train': timer.sum()}, epoch + 1)

		# save model
		torch.save(net.state_dict(), os.path.join(save_dir, f'./{log_id}-model-{epoch}.pth'))
		# save optim
		torch.save(optimizer.state_dict(), os.path.join(save_dir, f'./{log_id}-optim-{epoch}.pth'))

		# test!
		G.set('S', output_scale_S)
		G.set('B', 5)
		net.eval()
		metrics, timer = Accumulator(7), Timer()
		with torch.no_grad():
			timer.start()

			calc = metrics_utils.ObjectDetectionMetricsCalculator(G.get('num_classes'), 0.1)

			# test loss
			for i, (X, y) in enumerate(test_iter):
				X, y = X.to(devices[0]), [ys.to(devices[0]) for ys in y]
				yhat = net(X)
				calc.add_data(yhat, y)

				for j, (y_single, yhat_single) in enumerate(zip(y, yhat)):
					coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss = loss(yhat, y, 1000000) # very big epoch number to omit prior loss
					loss_val = coord_loss + class_loss + no_obj_loss + obj_loss + prior_loss
					metrics.add(coord_loss.sum(), class_loss.sum(), no_obj_loss.sum(), obj_loss.sum(), prior_loss.sum(), loss_val.sum(), X.shape[0])

					print(f'epoch {epoch} batch {i + 1}/{len(test_iter)} scale {G.get("scale")[j]} test loss: {metrics[5] / metrics[6]}, S: {G.get("S")}, B: {G.get("B")}')

			for j in range(num_scales + 1):
				log_loss_tensorboard(metrics, epoch, visualize_cnt, plot_indices, j, train=False)

			# log test mAP & PR Curve
			mAP = 0
			for c in range(G.get('num_classes')):
				pr_data = calc.calculate_precision_recall(0.5, c)
				p = torch.zeros(len(pr_data)) # precision
				r = torch.zeros(len(pr_data)) # recall
				z1 = torch.randint(0, len(pr_data), (len(pr_data),)) # dummy data
				z2 = torch.randint(0, len(pr_data), (len(pr_data),)) # dummy data
				z3 = torch.randint(0, len(pr_data), (len(pr_data),)) # dummy data
				z4 = torch.randint(0, len(pr_data), (len(pr_data),)) # dummy data
				for i, pr in enumerate(pr_data):
					p[i] = pr['precision']
					r[i] = pr['recall']
				pr_writer.add_pr_curve_raw(f'PR/{G.get("categories")[c]}', z1, z2, z3, z4, p, r, epoch + 1, len(pr_data))
				# calculate VOC mAP
				mAP += calc.calculate_average_precision(metrics_utils.InterpolationMethod.Interpolation_11, prl=pr_data)
			mAP /= G.get('num_classes')
			writer.add_scalars(f'mAP/VOC', {log_id: mAP}, epoch + 1)

			timer.stop()

			# log test timing
			writer.add_scalars(f'timing/{log_id}', {'test': timer.sum()}, epoch + 1)
