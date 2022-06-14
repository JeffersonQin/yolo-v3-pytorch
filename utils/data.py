import random
import math
from typing import Tuple
import warnings
import torch
import torchvision
from torch.utils import data
from . import globalvar as G


voc_categories = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', ]
coco_category_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
color_jitter = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)

def transform_crop(img: torch.Tensor, bbox: torch.Tensor, f: float) -> Tuple[torch.Tensor, torch.Tensor]:
	"""transformation: image cropping

	Args:
		img (torch.Tensor): image tensor
		bbox (torch.Tensor): [N, 5] absolute bbox tensor: [x1, y1, x2, y2, category], category starts from zero
		f (float): maximum factor of cropping off, 0.0 <= f < 1.0

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: return transformed image and bbox
	"""
	height = img.shape[1]
	width = img.shape[2]
	N = bbox.shape[0]

	# use random value to decide scaling factor on x and y axis
	random_height = random.random() * f
	random_width = random.random() * f
	# use random value again to decide scaling factor for 4 borders
	random_top = random.random() * random_height
	random_left = random.random() * random_width
	# calculate new width and height and position
	top = int(random_top * height)
	left = int(random_left * width)
	height = int(height - random_height * height)
	width = int(width - random_width * width)
	# crop image
	img = torchvision.transforms.functional.crop(img, top, left, height, width)

	bbox[:, 0] = torch.max(torch.zeros(N), bbox[:, 0] - left)
	bbox[:, 1] = torch.max(torch.zeros(N), bbox[:, 1] - top)
	bbox[:, 2] = torch.min(torch.ones(N) * width, bbox[:, 2] - left)
	bbox[:, 3] = torch.min(torch.ones(N) * height, bbox[:, 3] - top)

	return img, bbox


def transform_auto_padding(img: torch.Tensor, bbox: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""transformation: auto padding to square

	Args:
		img (torch.Tensor): image tensor
		bbox (torch.Tensor): [N, 5] absolute bbox tensor: [x1, y1, x2, y2, category], category starts from zero

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: return image and transformed bbox
	"""
	height = img.shape[1]
	width = img.shape[2]

	if height > width:
		left = int((height - width) / 2)
		top = 0
	else:
		left = 0
		top = int((width - height) / 2)

	# fill padding
	img = torchvision.transforms.functional.pad(img, (left, top))
	# transform bbox
	bbox[:, 0] = bbox[:, 0] + left
	bbox[:, 1] = bbox[:, 1] + top
	bbox[:, 2] = bbox[:, 2] + left
	bbox[:, 3] = bbox[:, 3] + top

	return img, bbox


def transform_random_padding(img: torch.Tensor, bbox: torch.Tensor, f: float) -> Tuple[torch.Tensor, torch.Tensor]:
	"""transformation: random padding to square for augmentation

	Args:
		img (torch.Tensor): image tensor
		bbox (torch.Tensor): [N, 5] absolute bbox tensor: [x1, y1, x2, y2, category], category starts from zero
		f (float): maximum factor of padding, f >= 0.0

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: return image and transformed bbox
	"""
	height = img.shape[1]
	width = img.shape[2]

	# use random value to decide scaling factor on x and y axis
	random_height = random.random() * f * height
	random_width = random.random() * f * width
	# use random value again to decide scaling factor for 4 borders
	random_top = int(random.random() * random_height)
	random_left = int(random.random() * random_width)
	random_bottom = int(random_height - random_top)
	random_right = int(random_width - random_left)

	# apply padding
	img = torchvision.transforms.functional.pad(img, (random_left, random_top, random_right, random_bottom))
	# transform bbox
	bbox[:, 0] = bbox[:, 0] + random_left
	bbox[:, 1] = bbox[:, 1] + random_top
	bbox[:, 2] = bbox[:, 2] + random_left
	bbox[:, 3] = bbox[:, 3] + random_top

	return img, bbox


def transform_horizontal_flip(img: torch.Tensor, bbox: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""transformation: horizontal flip

	Args:
		img (torch.Tensor): image tensor
		bbox (torch.Tensor): [N, 5] absolute bbox tensor: [x1, y1, x2, y2, category], category starts from zero

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: return image and transformed bbox
	"""
	width = img.shape[2]

	img = torchvision.transforms.functional.hflip(img)
	bbox_width = bbox[:, 2] - bbox[:, 0]
	bbox[:, 0] = width - bbox_width - bbox[:, 0]
	bbox[:, 2] = bbox[:, 0] + bbox_width

	return img, bbox


def transform_to_relative(img: torch.Tensor, bbox: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""transformation: transform bbox coordinates from absolute to relative

	Args:
		img (torch.Tensor): image tensor
		bbox (torch.Tensor): [N, 5] absolute bbox tensor: [x1, y1, x2, y2, category], category starts from zero

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: return image and transformed bbox
	"""
	height = img.shape[1]
	width = img.shape[2]

	bbox[:, 0] = bbox[:, 0] / width
	bbox[:, 1] = bbox[:, 1] / height
	bbox[:, 2] = bbox[:, 2] / width
	bbox[:, 3] = bbox[:, 3] / height

	return img, bbox


def transform_to_yolo(bbox: torch.Tensor, sf: int) -> torch.Tensor:
	"""transformation: transform bbox to yolo format

	Args:
		bbox (torch.Tensor): [N, 5] relative bbox tensor: [x1, y1, x2, y2, category], category starts from zero
		sf (int): final scale integer, equals to S (feature map) * scale (yolo head)

	Returns:
		torch.Tensor: YOLO format target
	"""
	B = G.get('B')
	num_classes = G.get('num_classes')

	label = torch.zeros((sf, sf, B * (5 + num_classes)))
	obj_cnt = torch.zeros((sf, sf))

	for box in bbox:
		x1, y1, x2, y2, cat = box[0], box[1], box[2], box[3], box[4]
		
		# In VOC dataset, some bboxes are tagged difficult
		# here if the category int is encoded slightly greater
		# then it is recognized as difficult
		cat_i = int(cat)
		if cat_i < cat: iou = 1.0000001
		else: iou = 1.0

		if x1 == x2 or y1 == y2: continue
		if x1 >= 1 or y1 >= 1 or x2 <= 0 or y2 <= 0: continue

		x = (x1 + x2) / 2
		y = (y1 + y2) / 2

		w = x2 - x1
		h = y2 - y1

		xidx = math.floor(x * sf)
		yidx = math.floor(y * sf)

		if obj_cnt[yidx][xidx] >= B:
			warnings.warn(f'More than {B} objects in one cell ({sf}x{sf})', RuntimeWarning, stacklevel=2)
			continue

		label[yidx][xidx][int(0 + (5 + num_classes) * obj_cnt[yidx][xidx])] = x * sf - xidx
		label[yidx][xidx][int(1 + (5 + num_classes) * obj_cnt[yidx][xidx])] = y * sf - yidx
		label[yidx][xidx][int(2 + (5 + num_classes) * obj_cnt[yidx][xidx])] = w
		label[yidx][xidx][int(3 + (5 + num_classes) * obj_cnt[yidx][xidx])] = h
		label[yidx][xidx][int(4 + (5 + num_classes) * obj_cnt[yidx][xidx])] = iou
		label[yidx][xidx][int(5 + (5 + num_classes) * obj_cnt[yidx][xidx] + cat_i)] = 1

		obj_cnt[yidx][xidx] += 1

	return label


class YOLODataset(data.Dataset):
	"""YOLO Dataset (base class)"""

	def __init__(self, dataset: data.Dataset, train: float=0.5):
		"""YOLO Dataset Initialization

		Args:
			dataset (data.Dataset): dataset
			train (float, optional): random ratio for data augmentation. Defaults to 0.5.
		"""
		self.dataset = dataset
		self.train = train


	def __len__(self):
		"""Get length of dataset

		Returns:
			int: length of dataset
		"""
		return len(self.dataset)


	def transform(self, img, target) -> Tuple[torch.Tensor, torch.Tensor]:
		"""transform (just an interface for performing transformation)
		transform image to torch.Tensor and target to bbox format
		[N, 5] absolute coordinates (x1, y1, x2, y2, category)
		category starts from zero.

		Args:
			img (Any): image
			target (Any): target

		Returns:
			Tuple[torch.Tensor, torch.Tensor]: converted bbox format data
		"""
		return img, target


	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Get item from dataset

		Args:
			idx (int): index of dataset

		Returns:
			Tuple[torch.Tensor, torch.Tensor]: (image, target)
		"""
		S = G.get('S')
		img, bbox = self.dataset[idx]
		img, bbox = self.transform(img, bbox)

		# Image Augmentation
		# randomly scaling and translation up to 20%
		if random.random() < self.train:
			img, bbox = transform_crop(img, bbox, 0.2)

		# color jitter
		# randomly adjust brightness, contrast, saturation, hue
		if random.random() < self.train:
			img = color_jitter(img)

		# randomly adjust padding up to 20%
		if random.random() < self.train:
			img, bbox = transform_random_padding(img, bbox, 0.2)

		# random horizontal flip
		if random.random() < self.train:
			img, bbox = transform_horizontal_flip(img, bbox)

		img, bbox = transform_auto_padding(img, bbox) # fix ratio
		img, bbox = transform_to_relative(img, bbox)
		img = torchvision.transforms.functional.resize(img, (S * 32, S * 32))

		# random noise
		if random.random() < self.train:
			img = img + torch.randn(img.size()) * 0.05

		# random gaussion blur
		if random.random() < self.train:
			img = torchvision.transforms.functional.gaussian_blur(img, 
				kernel_size=(random.randint(1, 3) * 2 - 1, random.randint(1, 3) * 2 - 1))

		labels = [transform_to_yolo(bbox, scale * S) for scale in G.get('scale')]
		return img, labels



class VOCDataset(YOLODataset):
	"""VOC Dataset"""

	def transform(self, img: torch.Tensor, target: dict) -> Tuple[torch.Tensor, torch.Tensor]:
		"""transform VOC data to YOLO data format

		Args:
			img (torch.Tensor): image
			target (dict): target

		Returns:
			Tuple[torch.Tensor, torch.Tensor]: image, bbox
		"""
		if not isinstance(target['annotation']['object'], list):
			target['annotation']['object'] = [target['annotation']['object']]
		count = len(target['annotation']['object'])

		bbox = torch.zeros((count, 5))
		for i in range(count):
			obj = target['annotation']['object'][i]
			bbox[i][0] = float(obj['bndbox']['xmin'])
			bbox[i][1] = float(obj['bndbox']['ymin'])
			bbox[i][2] = float(obj['bndbox']['xmax'])
			bbox[i][3] = float(obj['bndbox']['ymax'])
			bbox[i][4] = voc_categories.index(obj['name'])
			if obj['difficult'] == '1':
				bbox[i][4] += 0.1
		return img, bbox


class COCODataset(YOLODataset):
	"""COCO Dataset"""

	def transform(self, img: torch.Tensor, target: dict) -> Tuple[torch.Tensor, torch.Tensor]:
		"""transform COCO data to YOLO data format

		Args:
			img (torch.Tensor): image
			target (dict): target

		Returns:
			Tuple[torch.Tensor, torch.Tensor]: image, bbox
		"""
		count = len(target)
		bbox = torch.zeros((count, 5))
		for i in range(count):
			obj = target[i]
			bbox[i][0] = obj['bbox'][0]
			bbox[i][1] = obj['bbox'][1]
			bbox[i][2] = obj['bbox'][0] + obj['bbox'][2]
			bbox[i][3] = obj['bbox'][1] + obj['bbox'][3]
			bbox[i][4] = coco_category_index.index(obj['category_id'])
		return img, bbox


def load_data_voc(batch_size_train: int, batch_size_test: int, num_workers=0, persistent_workers=False, download=False, train_shuffle=True, test_shuffule=False, pin_memory=True, data_augmentation=0.5) -> list[data.DataLoader]:
	"""Load Pascal VOC dataset, consist of VOC2007trainval+test+VOC2012train, VOC2012val

	Args:
		batch_size_train (int): training batch size
		batch_size_test (int): testing batch_size
		num_workers (int, optional): number of workers. Defaults to 0.
		persistent_workers (bool, optional): persistent_workers. Defaults to False.
		download (bool, optional): whether to download. Defaults to False.
		train_shuffle (bool, optional): whether to shuffle train data. Defaults to True.
		test_shuffule (bool, optional): whether to shuffle test data. Defaults to False.
		pin_memory (bool, optional): whether to pin memory. Defaults to True.
		data_augmentation (float, optional): random ratio for data augmentation. Defaults to 0.5.

	Returns:
		list[data.DataLoader]: train_iter, test_iter
	"""
	trans = [ torchvision.transforms.ToTensor() ]
	trans = torchvision.transforms.Compose(trans)

	voc2007_trainval = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='trainval', download=download, transform=trans)
	voc2007_test = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='test', download=download, transform=trans)
	voc2012_train = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='train', download=download, transform=trans)
	voc2012_val = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='val', download=download, transform=trans)

	return (
		data.DataLoader(
			VOCDataset(data.ConcatDataset([voc2007_trainval, voc2007_test, voc2012_train]), train=data_augmentation), 
			batch_size=batch_size_train, shuffle=train_shuffle, num_workers=num_workers, 
			persistent_workers=persistent_workers, pin_memory=pin_memory),
		data.DataLoader(
			VOCDataset(voc2012_val, train=0),
			batch_size=batch_size_test, shuffle=test_shuffule, num_workers=num_workers, 
			persistent_workers=persistent_workers, pin_memory=pin_memory)
	)


def load_data_coco(batch_size_train: int, batch_size_test: int, num_workers=0, persistent_workers=False, train_shuffle=True, test_shuffule=False, pin_memory=True, data_augmentation=0.5) -> list[data.DataLoader]:
	"""Load MSCOCO Detection dataset, consist of COCO2017train(trainval35k) and COCO2017val

	Args:
		batch_size_train (int): training batch size
		batch_size_test (int): testing batch_size
		num_workers (int, optional): number of workers. Defaults to 0.
		persistent_workers (bool, optional): persistent_workers. Defaults to False.
		download (bool, optional): whether to download. Defaults to False.
		train_shuffle (bool, optional): whether to shuffle train data. Defaults to True.
		test_shuffule (bool, optional): whether to shuffle test data. Defaults to False.
		pin_memory (bool, optional): whether to pin memory. Defaults to True.
		data_augmentation (float, optional): random ratio for data augmentation. Defaults to 0.5.

	Returns:
		list[data.DataLoader]: train_iter, test_iter
	"""
	trans = [ torchvision.transforms.ToTensor() ]
	trans = torchvision.transforms.Compose(trans)

	coco2017_train = torchvision.datasets.CocoDetection(
		'../data/COCODetection/train2017', 
		'../data/COCODetection/annotations_trainval2017/annotations/instances_train2017.json', 
		transform=trans)
	coco2017_val = torchvision.datasets.CocoDetection(
		'../data/COCODetection/val2017', 
		'../data/COCODetection/annotations_trainval2017/annotations/instances_val2017.json', 
		transform=trans)

	return (
		data.DataLoader(COCODataset(coco2017_train, train=data_augmentation),
			batch_size=batch_size_train, shuffle=train_shuffle, num_workers=num_workers, 
			persistent_workers=persistent_workers, pin_memory=pin_memory),
		data.DataLoader(COCODataset(coco2017_val, train=0), 
			batch_size=batch_size_test, shuffle=test_shuffule, num_workers=num_workers, 
			persistent_workers=persistent_workers, pin_memory=pin_memory)
	)
