import os
import random
import sys
import traceback
from typing import Optional
import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard

from torch.utils.data import DataLoader
from utils import metrics as metrics_utils
from utils.utils import Accumulator, Timer, get_all_gpu, update_lr
from utils import G
from yolo.loss import YoloLoss


def train(get_net, train_iter: DataLoader, test_iter: DataLoader, num_epochs: int, freeze_epoch: int, multi_scale_epoch: int, output_scale_S: int, lambda_scale: list[float], conf_thres: float, conf_ratio_thres: float, lr, get_optimizer, log_id: str, loss=YoloLoss(), num_gpu: int=1, accum_batch_num: int=1, mix_precision: bool=True, grad_clip: bool=True, clip_max_norm: float=5.0, model_dir: str='./model', log_dir: str='./logs', load_model: Optional[str]=None, load_optim: Optional[str]=None, load_epoch: int=-1, visualize_cnt: int=10, test_pr_batch_ratio: float=1.0, test_pr_after_epoch: int=0, skip_nan_inf: bool=False, auto_restore: bool=True, cloud_notebook_service: bool=False):
	"""trainer for yolo v2. 
	Note: weight init is not done in this method, because the architecture
	of yolo v2 is rather complicated with the design of pass through layer

	Args:
		get_net: module network getter
		train_iter (DataLoader): training dataset iterator
		test_iter (DataLoader): testing dataset iterator
		num_epochs (int): number of epochs to train
		freeze_epoch (int): epoch to freeze the backbone
		multi_scale_epoch (int): number of epochs to train with multi scale
		output_scale_S (int): final network scale (S), input size will be 32S * 32S, as the network stride is 32
		lambda_scale (list[float]): lambda list for each scale
		conf_thres (float): confidence threshold when calculating (m)AP
		conf_ratio_thres (float): confidence ratio threshold when calculating (m)AP
		lr (float | callable): learning rate or learning rate scheduler function relative to epoch
		get_optimizer: optimizer getter
		log_id (str): identifier for logging in tensorboard.
		loss (YoloLoss): loss function
		num_gpu (int, optional): number of gpu to train on, used for parallel training. Defaults to 1.
		accum_batch_num (int, optional): number of batch to accumulate gradient, used to solve OOM problem when using big batch sizes. Defaults to 1.
		mix_precision (bool, optional): whether to use mix_precision. Defaults to True.
		grad_clip (bool, optional): whether to use gradient clipping. Defaults to True.
		clip_max_norm (float, optional): max_norm when gradient clipping is used. Defaults to 5.0.
		model_dir (str, optional): saving directory for model weights. Defaults to './model'.
		log_dir (str, optional): saving directory for tensorboard logs. Defaults to './logs'.
		load_model (Optional[str], optional): path of model weights to load if exist. Defaults to None.
		load_optim (Optional[str], optional): path of optimizer state_dict to load if exist. Defaults to None.
		load_epoch (int, optional): done epoch count minus one when loading, should be the same with the number in auto-saved file name. Defaults to -1.
		visualize_cnt (int, optional): number of batches to visualize each epoch during training progress. Defaults to 10.
		test_pr_batch_ratio (float, optional): ratio of batches to test average precision each epoch. Default to 1.0, that is all batches.
		test_pr_after_epoch (int, optional): test average precision after number of epoch. Defaults to 0.
		skip_nan_inf (bool, optional): whether to skip nan and inf in loss. Defaults to False.
		auto_restore (bool, optional): whether to restore model and optimizer state_dict after nan/inf or exception occurred. Defaults to True.
		cloud_notebook_service (bool, optional): whether to using cloud notebook services such as Kaggle, Colab, etc. Defaults to False.
	"""
	os.makedirs(model_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	net = get_net()
	optimizer = get_optimizer(net)

	# tensorboard
	writer = tensorboard.SummaryWriter(os.path.join(log_dir, 'yolo'))
	pr_writer = tensorboard.SummaryWriter(os.path.join(log_dir, f'yolo/pr/{log_id}'))

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

	if mix_precision:
		scaler = torch.cuda.amp.GradScaler()

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


	def load_model_and_optim(epoch: int, net):
		try:
			net.to('cpu')
			# free memory
			if torch.cuda.is_available():
				for _ in range(10):
					torch.cuda.empty_cache()
			# redefine net and optimizer
			net = get_net()
			optimizer = get_optimizer(net)
			# load model and optim
			net.load_state_dict(torch.load(os.path.join(model_dir, f'./{log_id}-model-{epoch}.pth')))
			net.to(devices[0])
			optimizer.load_state_dict(torch.load(os.path.join(model_dir, f'./{log_id}-optim-{epoch}.pth')))
			# free memory
			if torch.cuda.is_available():
				for _ in range(10):
					torch.cuda.empty_cache()
			return net, optimizer
		except Exception as e:
			log_alert(f'[Exception occurred] epoch: {epoch}, {repr(e)}')
			log_alert(f'exc: {traceback.format_exc()}')
			log_alert(f'stack: {traceback.format_stack}')
			log_alert('load model and optim failed')
			raise e


	def log_alert(msg: str):
		print(msg)
		with open(os.path.join(model_dir, f'./{log_id}-alert.txt'), 'a+') as f:
			f.write(f'{msg}\n')


	def log_results(msg: str):
		print(msg)
		with open(os.path.join(model_dir, f'./{log_id}-results.txt'), 'a+') as f:
			f.write(f'{msg}\n')


	if cloud_notebook_service:
		import ipywidgets as widgets
		from IPython.display import FileLink, display
		ds_widgets = {
			'status': widgets.Textarea(
				value='',
				placeholder='',
				description='status',
				disabled=True
			),
			'result': widgets.Textarea(
				value='',
				placeholder='',
				description='result',
				disabled=True
			),
		}
		for _, v in ds_widgets.items():
			display(v)
		log_alert('')
		log_results('')
		display(FileLink((os.path.join(model_dir, f'./{log_id}-alert.txt'))))
		display(FileLink((os.path.join(model_dir, f'./{log_id}-results.txt'))))


	def display_text(msg: str, tag: str):
		if cloud_notebook_service:
			ds_widgets[tag].value = msg
		else: print(msg)


	def main_loop(epoch):
		for param in net.backbone.parameters():
			if epoch < freeze_epoch:
				param.requires_grad = False
			else:
				param.requires_grad = True

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
			with torch.autocast(devices[0].type, enabled=mix_precision):
				yhat = net(X)
				plot_indices = plot(i + 1, num_batches, visualize_cnt)
				total_loss = 0
				status_str = ''

				for j, (y_single, yhat_single) in enumerate(zip(y, yhat)):
					coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss = loss(yhat_single, y_single, epoch)
					no_obj_loss = no_obj_loss * lambda_scale[j]
					prior_loss = prior_loss * lambda_scale[j]
					loss_val = coord_loss + class_loss + no_obj_loss + obj_loss + prior_loss

					# if NaN, do not affect training loop
					if not (torch.isnan(loss_val.sum()) or torch.isinf(loss_val.sum())):
						total_loss = total_loss + loss_val.sum()

						with torch.no_grad():
							metrics[j].add(coord_loss.sum(), class_loss.sum(), no_obj_loss.sum(), obj_loss.sum(), prior_loss.sum(), loss_val.sum(), X.shape[0])
							metrics[num_scales].add(coord_loss.sum(), class_loss.sum(), no_obj_loss.sum(), obj_loss.sum(), prior_loss.sum(), loss_val.sum(), 0)
					else:
						loss_alert = f'[NaN/Inf occurred] epoch: {epoch}, batch: {i}, coord_loss: {float(coord_loss.sum())}, class_loss: {float(class_loss.sum())}, no_obj_loss: {float(no_obj_loss.sum())}, obj_loss: {float(obj_loss.sum())}, prior_loss: {float(prior_loss.sum())}'
						log_alert(loss_alert)
						if skip_nan_inf:
							metrics[num_scales].add(0, 0, 0, 0, 0, 0, -X.shape[0])
						else:
							raise Exception(loss_alert)

					# log train loss
					status_str = status_str + f'epoch {epoch} batch {i + 1}/{num_batches} scale {G.get("scale")[j]} loss: {metrics[j][5] / metrics[j][6]}, S: {G.get("S")}, B: {G.get("B")}\n'
					if plot_indices > 0:
						log_loss_tensorboard(metrics, epoch, visualize_cnt, plot_indices, j, train=True)

				# log total scale loss
				metrics[num_scales].add(0, 0, 0, 0, 0, 0, X.shape[0])
				status_str = status_str + f'epoch {epoch} batch {i + 1}/{num_batches} total loss: {metrics[num_scales][5] / metrics[num_scales][6]}, S: {G.get("S")}, B: {G.get("B")}'
				display_text(status_str, 'status')

				if plot_indices > 0:
					log_loss_tensorboard(metrics, epoch, visualize_cnt, plot_indices, num_scales, train=True)

			# backward to accumulate gradients
			if mix_precision:
				scaler.scale(total_loss.sum()).backward()
			else: 
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
				if mix_precision:
					if grad_clip:
						scaler.unscale_(optimizer)
						torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_max_norm)
					scaler.step(optimizer)
					scaler.update()
				else:
					if grad_clip:
						torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_max_norm)
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
		torch.save(net.state_dict(), os.path.join(model_dir, f'./{log_id}-model-{epoch}.pth'))
		# save optim
		torch.save(optimizer.state_dict(), os.path.join(model_dir, f'./{log_id}-optim-{epoch}.pth'))

		if cloud_notebook_service:
			from IPython.display import FileLink, display
			display(FileLink((os.path.join(model_dir, f'./{log_id}-model-{epoch}.pth'))))
			display(FileLink((os.path.join(model_dir, f'./{log_id}-optim-{epoch}.pth'))))

		# test!
		G.set('S', output_scale_S)
		net.eval()
		metrics, timer = [Accumulator(7) for _ in range(num_scales + 1)], Timer()
		with torch.no_grad():
			timer.start()

			calc = metrics_utils.ObjectDetectionMetricsCalculator(G.get('num_classes'), conf_thres, conf_ratio_thres)

			# test loss
			for i, (X, y) in enumerate(test_iter):
				X, y = X.to(devices[0]), [ys.to(devices[0]) for ys in y]
				yhat = net(X)
				status_str = ''

				if i < len(test_iter) * test_pr_batch_ratio and epoch >= test_pr_after_epoch:
					calc.add_data(yhat, y[0])

				for j, (y_single, yhat_single) in enumerate(zip(y, yhat)):
					coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss = loss(yhat_single, y_single, 1000000) # very big epoch number to omit prior loss
					no_obj_loss = no_obj_loss * lambda_scale[j]
					prior_loss = prior_loss * lambda_scale[j]
					loss_val = coord_loss + class_loss + no_obj_loss + obj_loss + prior_loss

					if (torch.isnan(loss_val.sum()) or torch.isinf(loss_val.sum())) and auto_restore:
						loss_alert = f'[NaN/Inf occurred] epoch: {epoch}, batch: {i}, coord_loss: {float(coord_loss.sum())}, class_loss: {float(class_loss.sum())}, no_obj_loss: {float(no_obj_loss.sum())}, obj_loss: {float(obj_loss.sum())}, prior_loss: {float(prior_loss.sum())}'
						log_alert(loss_alert)
						raise Exception(loss_alert)

					metrics[j].add(coord_loss.sum(), class_loss.sum(), no_obj_loss.sum(), obj_loss.sum(), prior_loss.sum(), loss_val.sum(), X.shape[0])
					metrics[num_scales].add(coord_loss.sum(), class_loss.sum(), no_obj_loss.sum(), obj_loss.sum(), prior_loss.sum(), loss_val.sum(), 0)

					status_str = status_str + f'epoch {epoch} batch {i + 1}/{len(test_iter)} scale {G.get("scale")[j]} test loss: {metrics[j][5] / metrics[j][6]}, S: {G.get("S")}, B: {G.get("B")}\n'
				metrics[num_scales].add(0, 0, 0, 0, 0, 0, X.shape[0])
				display_text(status_str, 'status')

			for j in range(num_scales + 1):
				log_loss_tensorboard(metrics, epoch + 1, visualize_cnt, 0, j, train=False)

			if epoch >= test_pr_after_epoch:
				# log test mAP & PR Curve
				mAP5 = 0
				mAPVOC = 0
				for c in range(G.get('num_classes')):
					pr_data = calc.calculate_precision_recall(0.5, c)
					if len(pr_data) <= 0: continue
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
					# calculate COCO mAP AP@.5
					mAP5 += calc.calculate_average_precision(metrics_utils.InterpolationMethod.Interpolation_101, prl=pr_data)
					# calculate VOC mAP
					mAPVOC += calc.calculate_average_precision(metrics_utils.InterpolationMethod.Interpolation_11, prl=pr_data)
				mAP5 /= G.get('num_classes')
				mAPVOC /= G.get('num_classes')
				result_str = f'epoch {epoch + 1} test mAP@.5: {mAP5}, VOCmAP: {mAPVOC}'
				log_results(result_str)
				display_text(result_str, 'result')
				writer.add_scalars(f'mAP/AP@.5-random-part', {log_id: mAP5}, epoch + 1)
				writer.add_scalars(f'mAP/VOCmAP-random-part', {log_id: mAPVOC}, epoch + 1)

			timer.stop()

			# log test timing
			writer.add_scalars(f'timing/{log_id}', {'test': timer.sum()}, epoch + 1)


	# train
	epoch = load_epoch
	while epoch + 1 < num_epochs:
		epoch += 1

		try:
			main_loop(epoch)
		except Exception as e:
			log_alert(f'[Exception occurred] epoch: {epoch}, {repr(e)}')
			log_alert(f'exc: {traceback.format_exc()}')
			log_alert(f'stack: {traceback.format_stack}')
			if auto_restore:
				log_alert(f'[auto restore] from {epoch} to {epoch - 1}')
				epoch -= 1
				net, optimizer = load_model_and_optim(epoch, net)
				continue

