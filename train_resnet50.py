import random
import numpy
import torch
from utils import G
from utils import data
from yolo.loss import YoloLoss
from yolo.model import ResNet50YoloDetector
from yolo.train import train

seed = 0
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
numpy.random.seed(seed)

# init global variables
G.init()

# define hyper parameters
batch_size_train = 8
batch_size_test = 64
accum_batch_num = 8
num_epoch = 160
multi_scale_epoch = 150
output_scale_S = 13
weight_decay = 0.0005
momentum = 0.9
clip_max_norm = 100.0
test_pr_after_epoch = 30
test_pr_batch_ratio = 1.0

# learning rate scheduler
def lr(epoch):
	if epoch < 10: return 0.00001 * (epoch + 1)
	if epoch < 20: return 0.0001 * (epoch - 9)
	if epoch < 60: return 0.001
	if epoch < 105: return 0.0001
	return 0.00001

loss = YoloLoss()
loss.cfg_yolov3()


if __name__ == '__main__':
	# data loader
	train_iter, test_iter = data.load_data_coco(batch_size_train, batch_size_test, train_shuffle=True, test_shuffule=True, data_augmentation=True)

	# define network
	detector = ResNet50YoloDetector()

	# weight init
	detector.winit()

	optimizer = torch.optim.SGD(detector.parameters(), lr=lr(0), weight_decay=weight_decay, momentum=momentum)

	# train
	train(
		detector, train_iter, test_iter,
		num_epochs=num_epoch,
		multi_scale_epoch=multi_scale_epoch,
		output_scale_S=output_scale_S,
		lr=lr,
		optimizer=optimizer,
		log_id=f'resnet50-sgd-{output_scale_S * 32}',
		loss=loss,
		num_gpu=1,
		accum_batch_num=accum_batch_num,
		mix_precision=True,
		grad_clip=True,
		clip_max_norm=clip_max_norm,
		save_dir='./model',
		load_model='./model/resnet50-sgd-416-model-19.pth',
		load_optim='./model/resnet50-sgd-416-optim-19.pth',
		load_scaler='./model/resnet50-sgd-416-scaler-19.pth',
		load_epoch=19,
		visualize_cnt=100,
		test_pr_batch_ratio=test_pr_batch_ratio,
		test_pr_after_epoch=test_pr_after_epoch
	)
