import os
import random
import time
import traceback
import numpy
import torch
from utils import G
from utils import data
from utils.utils import CosineAnnealingScheduler, LearningRateSchedulerComposer, LinearWarmupScheduler
from yolo.loss import YoloLoss
from yolo.model import ResNet18YoloDetector
from yolo.train import train

seed = 31
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
numpy.random.seed(seed)

# init global variables
G.init()
G.set('categories', [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', ])
G.set('num_classes', 20)
G.set('colors', [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)])

# define hyper parameters
# batch & gradient accumulation
batch_size_train = 32
batch_size_test = 64
accum_batch_num = 2
# epoch
num_epoch = 200
freeze_epoch = 30
multi_scale_epoch = 190
output_scale_S = 13
# optimizer
weight_decay = 0.0005
momentum = 0.9
# mix precision
mix_precision = True
# gradient clipping
clip_max_norm = 20.0
# lambda scale
lambda_scale_1 = 1
lambda_scale_2 = 1
lambda_scale_4 = 1
# loss
lambda_coord = 10.0
lambda_noobj = 1.0
lambda_obj = 10.0
lambda_class = 10.0
lambda_prior = 0.1
epoch_prior = 60
IoU_thres = 0.7
scale_coord = True
eps = 1e-6
no_obj_v3 = True
# learning rate scheduler
lr_linear_max = 0.15
lr_warmup_epoch = 30
lr_cosine_max_1 = 0.15
lr_T_half_1 = 170
# lr_cosine_max_2 = 0.5
# lr_T_half_2 = 50
# lr_cosine_max_3 = 0.25
# lr_T_half_3 = 60
# conf thres
conf_thres = 0.01
conf_ratio_thres = 0.05
# data augmentation
random_ratio = 0.5
# test strategy
test_pr_after_epoch = 10
test_pr_batch_ratio = 1.0

log_id = f'resnet18-voc-no-crop-sgd-{output_scale_S * 32}'
load_epoch = -1
model_dir = './model'
log_dir = './logs'

lr = LearningRateSchedulerComposer([
	LinearWarmupScheduler(lr_linear_max, lr_warmup_epoch),
	CosineAnnealingScheduler(lr_cosine_max_1, lr_T_half_1),
	# CosineAnnealingScheduler(lr_cosine_max_2, lr_T_half_2),
	# CosineAnnealingScheduler(lr_cosine_max_3, lr_T_half_3),
])

loss = YoloLoss(
	lambda_coord=lambda_coord,
	lambda_noobj=lambda_noobj,
	lambda_obj=lambda_obj,
	lambda_class=lambda_class,
	lambda_prior=lambda_prior,
	IoU_thres=IoU_thres,
	epoch_prior=epoch_prior,
	scale_coord=scale_coord,
	no_obj_v3=no_obj_v3,
	eps=eps
)


if __name__ == '__main__':
	# data loader
	train_iter, test_iter = data.load_data_voc(batch_size_train, batch_size_test, train_shuffle=True, test_shuffule=True, data_augmentation=random_ratio)

	# define network
	def detector():
		net = ResNet18YoloDetector()
		# init weight
		net.winit()
		return net

	def optimizer(net):
		return torch.optim.SGD(net.parameters(), lr=lr(0), weight_decay=weight_decay, momentum=momentum)

	try:
		# train
		train(
			detector, train_iter, test_iter,
			num_epochs=num_epoch,
			freeze_epoch=freeze_epoch,
			multi_scale_epoch=multi_scale_epoch,
			output_scale_S=output_scale_S,
			lambda_scale=[lambda_scale_1, lambda_scale_2, lambda_scale_4],
			conf_thres=conf_thres,
			conf_ratio_thres=conf_ratio_thres,
			lr=lr,
			get_optimizer=optimizer,
			log_id=log_id,
			loss=loss,
			num_gpu=1,
			accum_batch_num=accum_batch_num,
			mix_precision=mix_precision,
			grad_clip=True,
			clip_max_norm=clip_max_norm,
			model_dir=model_dir,
			log_dir=log_dir,
			# load_model=os.path.join(model_dir, f'{log_id}-model-{load_epoch}.pth'),
			# load_optim=os.path.join(model_dir, f'{log_id}-optim-{load_epoch}.pth'),
			# load_model='./model/resnet18-voc-sgd-multi-scale-epoch-130-model.pth',
			# load_optim='./model/resnet18-voc-sgd-multi-scale-epoch-130-optim.pth',
			# load_model='./model/resnet18-voc-double-cosine-sgd-416-model-199.pth',
			# load_optim='./model/resnet18-voc-double-cosine-sgd-416-optim-199.pth',
			# load_epoch=load_epoch,
			visualize_cnt=10,
			test_pr_batch_ratio=test_pr_batch_ratio,
			test_pr_after_epoch=test_pr_after_epoch,
			skip_nan_inf=False,
			auto_restore=True,
			cloud_notebook_service=False
		)
	except Exception as e:
		print(repr(e))
		traceback.print_exc()
		print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )

		from playsound import playsound
		while True:
			playsound('./assets/radar.mp3')
