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


def linear_warmup_cosine_lr_scheduler(linear_max, warmup_epoch, T_half):
	def lr(epoch):
		if epoch < warmup_epoch: return linear_max / warmup_epoch * (epoch + 1)
		epoch = (epoch - warmup_epoch) % T_half
		return linear_max * 0.5 * (1 + cos(epoch / T_half * pi))
	return lr


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
