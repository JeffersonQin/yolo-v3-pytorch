import random
import cv2
import torch
from utils import data, metrics
from utils import G
import utils
from utils import visualize
from utils.utils import Timer
from utils.visualize import tensor_to_cv2
from yolo.converter import Yolo2BBox, Yolo2BBoxSingle
from yolo.model import ResNet18YoloDetector
from yolo.nms import YoloNMS

G.init()
G.set('S', 13)
G.set('categories', [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', ])
G.set('num_classes', 20)
G.set('colors', [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)])

# a = data.load_data_voc(10)
# i = 1
# for x in a[0]:
# 	# print(len(x))
# 	X, Y = x
# 	print(X.shape)
# 	print(len(Y))
# 	print(Y[0].shape)
# 	print(Y[1].shape)
# 	print(Y[2].shape)
# 	i += 1
# 	if i > 10: break
net = ResNet18YoloDetector()
net.load_state_dict(torch.load('./model/resnet18-voc-adaptive-test-sgd-416-model-20.pth'))
net.to('cuda')
net.eval()
train_iter, test_iter = data.load_data_voc(64, 64)
with torch.no_grad():
	calc = metrics.ObjectDetectionMetricsCalculator(G.get('num_classes'), 0.001, 0.2)
	timer = Timer()
	timer.start()
	for i, (X, y) in enumerate(test_iter):
		print(f'batch {i+1}/{len(test_iter)}')
		X, y = X.to('cuda'), [ys.to('cuda') for ys in y]
		yhat = net(X)
		calc.add_data(yhat, y[0])
	timer.stop()
	mAP = calc.calculate_COCOmAP50()
	print(f'COCOmAP:{mAP}')
	mAP = calc.calculate_VOCmAP()
	print(f'VOCmAP:{mAP}')
	print(f'time: {timer.sum()}')

# with torch.no_grad():
# 	for i, (X, y) in enumerate(test_iter):
# 		X, y = X.to('cuda'), [ys.to('cuda') for ys in y]
# 		yhat = net(X)
# 		converter = Yolo2BBox()
# 		single_converter = Yolo2BBoxSingle()
# 		nms = YoloNMS()
# 		Yhatbbox = converter(yhat)
# 		Ybbox = single_converter(y[0])
# 		j=1
# 		for x, yhatbbox, y in zip(X, Yhatbbox, Ybbox):
# 			yhatbbox = nms(yhatbbox)
# 			img = visualize.draw_detection_result(visualize.tensor_to_cv2(x), yhatbbox, thres=0.1)
# 			cv2.imwrite(f'test-{j}.png', img)
# 			img = visualize.draw_detection_result(visualize.tensor_to_cv2(x), y, thres=0.1)
# 			cv2.imwrite(f'truth-{j}.png', img)
# 			j+=1
# 			if j>10:break
# 		Yhatbbox1 = single_converter(yhat[0])
# 		Yhatbbox2 = single_converter(yhat[1])
# 		Yhatbbox4 = single_converter(yhat[2])
# 		j=1
# 		for x, yhatbbox1, yhatbbox2, yhatbbox4 in zip(X, Yhatbbox1, Yhatbbox2, Yhatbbox4):
# 			yhatbbox = nms(yhatbbox)
# 			img = visualize.draw_detection_result(visualize.tensor_to_cv2(x), yhatbbox1, thres=0.1)
# 			cv2.imwrite(f'test-{j}-scale-1.png', img)
# 			img = visualize.draw_detection_result(visualize.tensor_to_cv2(x), yhatbbox2, thres=0.1)
# 			cv2.imwrite(f'test-{j}-scale-2.png', img)
# 			img = visualize.draw_detection_result(visualize.tensor_to_cv2(x), yhatbbox4, thres=0.1)
# 			cv2.imwrite(f'test-{j}-scale-4.png', img)
# 			j+=1
# 			if j>10:break
# 		break
