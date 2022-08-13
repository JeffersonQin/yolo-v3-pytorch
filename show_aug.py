import cv2
import torchvision
import random
from utils import visualize
from torch.utils import data

from utils.data import VOCDataset
from utils import G
from yolo.converter import Yolo2BBox, Yolo2BBoxSingle

G.init()
G.set('S', 13)
G.set('categories', [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', ])
G.set('num_classes', 20)
G.set('colors', [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)])
trans = [ torchvision.transforms.ToTensor() ]
trans = torchvision.transforms.Compose(trans)
voc2007_trainval = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='trainval', download=False, transform=trans)
voc2007_test = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='test', download=False, transform=trans)
voc2012_train = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='train', download=False, transform=trans)
voc2012_val = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='val', download=False, transform=trans)
dataset = VOCDataset(data.ConcatDataset([voc2007_trainval, voc2007_test, voc2012_train]), 0.5)
converter = Yolo2BBoxSingle()

for i in range(10):
	img, bbox = dataset[216]
	bbox = converter(bbox[0])
	img = visualize.draw_detection_result(visualize.tensor_to_cv2(img), bbox, thres=0.1)
	cv2.imshow(f'test{i}', img)

cv2.waitKey(0)
