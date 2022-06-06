from math import cos, pi
import torch
import numpy as np
import time


__all__ = ['try_gpu', 'Accumulator', 'Timer']


def try_gpu():
	if torch.cuda.is_available():
		return torch.device('cuda')
	else:
		return torch.device('cpu')


def get_all_gpu(num_gpu: int):
	return [torch.device('cuda:' + str(i)) for i in range(num_gpu)]


def update_lr(opt: torch.optim.Optimizer, lr: float):
	"""update learning rate for all parameters

	Args:
		opt (torch.optim.Optimizer): optimizer
		lr (float): learning rate
	"""
	for param_group in opt.param_groups:
		param_group['lr'] = lr


def test_trigger():
	"""test trigger, triggered when file './test_trigger' contains something"""
	try:
		with open('./test_trigger', 'r') as f:
			c = f.read()
			if len(c) > 0:
				return True
		return False
	except:
		return False


class LearningRateScheduler:
	def __init__(self, lr_scheduler, max_epoch):
		self.lr_scheduler = lr_scheduler
		self.max_epoch = max_epoch


	def __call__(self, epoch):
		return self.lr_scheduler(epoch)


class LinearWarmupScheduler(LearningRateScheduler):
	def __init__(self, linear_max, max_epoch):
		def lr(epoch):
			return linear_max / max_epoch * (epoch + 1)
		super().__init__(lr, max_epoch)


class CosineAnnealingScheduler(LearningRateScheduler):
	def __init__(self, cosine_max, T_half):
		def lr(epoch):
			epoch = epoch % T_half
			return cosine_max * 0.5 * (1 + cos(epoch / T_half * pi))
		super().__init__(lr, T_half)


class LearningRateSchedulerComposer:
	def __init__(self, lr_schedulers: list[LearningRateScheduler]):
		self.lr_schedulers = lr_schedulers


	def __call__(self, epoch):
		for lr_scheduler in self.lr_schedulers:
			if epoch < lr_scheduler.max_epoch:
				return lr_scheduler(epoch)
			else: epoch -= lr_scheduler.max_epoch
		return 0.0


# from d2l
class Accumulator(object):
    """
    Sum a list of numbers over time
    from: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/base.py
    """
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [float(a) + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0] * len(self.data)
    def __getitem__(self, i):
        return float(self.data[i])


class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer and record the time in a list"""
        self.times.append(time.time() - self.start_time)
        return self.times[-1]
        
    def avg(self):
        """Return the average time"""
        return sum(self.times)/len(self.times)
    
    def sum(self):
        """Return the sum of time"""
        return sum(self.times)
        
    def cumsum(self):
        """Return the accumuated times"""
        return np.array(self.times).cumsum().tolist()
