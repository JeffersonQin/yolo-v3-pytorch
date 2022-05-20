import json
import random
import torch


__all__ = ['init', 'set', 'get']


def init(S=19, B=3):
	"""Init the global variables"""
	global global_dict
	global_dict = {}
	# init values
	# MSCOCO categories
	global_dict['categories'] = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', ]
	# VOC categories
	# global_dict['categories'] = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', ]
	global_dict['num_classes'] = len(global_dict['categories'])
	global_dict['colors'] = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(global_dict['num_classes'])]
	global_dict['S'] = S
	global_dict['B'] = B
	global_dict['scale'] = [1, 2, 4]
	# init anchors
	with open(f'anchors/anchors-{B * len(global_dict["scale"])}.json', 'r', encoding='utf-8') as f:
		anchors = json.load(f)
	t_anchors = torch.zeros(len(anchors), 2)
	areas = torch.zeros(len(anchors))
	for i in range(len(anchors)):
		t_anchors[i][0] = anchors[i]['width']
		t_anchors[i][1] = anchors[i]['height']
		areas[i] = anchors[i]['width'] * anchors[i]['height']
	anchor_idx =  torch.argsort(areas, descending=True)
	global_dict['anchors'] = t_anchors[anchor_idx].reshape(len(global_dict["scale"]), B, 2)


def set(key, val):
	"""Set value

	Args:
		key (Any): key
		val (Any): value
	"""
	global_dict[key] = val


def get(key):
	"""Get value

	Args:
		key (Any): key

	Returns:
		Any: value
	"""
	return global_dict[key]
