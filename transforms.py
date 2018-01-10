from __future__ import print_function

import numpy as np 

# scipy.ndimage -> skimage -> cv2, 
# skimage is one or two orders of magnitude slower than cv2
import cv2

try:
	import torch
except ImportError:
	pass

import collections
import numbers
import types



InterpolationFlags = {'nearest':cv2.INTER_NEAREST, 'linear':cv2.INTER_LINEAR, 
				 	  'cubic':cv2.INTER_CUBIC, 'area':cv2.INTER_AREA, 
				 	  'lanczos':cv2.INTER_LANCZOS4}

BorderTypes = {'constant':cv2.BORDER_CONSTANT, 
			   'replicate':cv2.BORDER_REPLICATE, 'nearest':cv2.BORDER_REPLICATE,
			   'reflect':cv2.BORDER_REFLECT, 'mirror': cv2.BORDER_REFLECT,
			   'wrap':cv2.BORDER_WRAP, 'reflect_101':cv2.BORDER_REFLECT_101,}



def _loguniform(interval, random_state=np.random):
	low, high = interval
	return np.exp(random_state.uniform(np.log(low), np.log(high)))


def _clamp(img, min=None, max=None, dtype='uint8'):
	if min is None and max is None:
		if dtype == 'uint8':
			min, max = 0, 255
		elif dtype == 'uint16':
			min, max = 0, 65535
		else:
			min, max = -np.inf, np.inf
	img = np.clip(img, min, max)
	return img.astype(dtype)


def _jaccard(boxes, rect):
	def _intersect(boxes, rect):
		lt = np.maximum(boxes[:, :2], rect[:2])
		rb = np.minimum(boxes[:, 2:], rect[2:])
		inter = np.clip(rb - lt, 0, None)
		return inter[:, 0] * inter[:, 1]

	inter = _intersect(boxes, rect)
	
	area1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
	area2 = (rect[2] - rect[0]) * (rect[3] - rect[1])
	union = area1 + area2 - inter 
	
	jaccard  = inter / np.clip(union, 1e-10, None)
	coverage = inter / np.clip(area1, 1e-10, None)
	return jaccard, coverage, inter


def _coords_clamp(cds, shape, outside=None):
	w, h = shape[1] - 1, shape[0] - 1
	if outside == 'discard':
		cds_ = []
		for x, y in cds:
			x_ = x if 0 <= x <= w else np.sign(x) * np.inf
			y_ = y if 0 <= y <= h else np.sign(y) * np.inf
			cds_.append([x_, y_])
		return np.array(cds_, dtype=np.float32)
	else:
		return np.array([[np.clip(cd[0], 0, w), np.clip(cd[1], 0, h)] for cd in cds], dtype=np.float32)


def _to_bboxes(cds, img_shape=None):
	assert len(cds) % 4 == 0

	h, w = img_shape if img_shape is not None else (np.inf, np.inf)
	boxes = []
	cds = np.array(cds)
	for i in range(0, len(cds), 4):
		xmin = np.clip(cds[i:i+4, 0].min(), 0, w - 1)
		xmax = np.clip(cds[i:i+4, 0].max(), 0, w - 1)
		ymin = np.clip(cds[i:i+4, 1].min(), 0, h - 1)
		ymax = np.clip(cds[i:i+4, 1].max(), 0, h - 1)
		boxes.append([xmin, ymin, xmax, ymax])
	return np.array(boxes)


def _to_coords(boxes):
	cds = []
	for box in boxes:
		xmin, ymin, xmax, ymax = box 
		cds += [
			[xmin, ymin],
			[xmax, ymin],
			[xmax, ymax],
			[xmin, ymax],
		]
	return np.array(cds)


# recursively reset transform's state
def transform_state(t, **kwargs):
	if callable(t):
		t_vars = vars(t)

		if 'random_state' in kwargs and 'random' in t_vars:
			t.__dict__['random'] = kwargs['random_state']

		support = ['fillval', 'anchor', 'prob', 'mean', 'std', 'outside']
		for arg in kwargs:
			if arg in t_vars and arg in support:
				t.__dict__[arg] = kwargs[arg]

		if 'mode' in kwargs and 'mode' in t_vars:
			t.__dict__['mode'] = kwargs['mode']
		if 'border' in kwargs and 'border' in t_vars:
			t.__dict__['border'] = BorderTypes.get(kwargs['border'], cv2.BORDER_REPLICATE)

		if 'transforms' in t_vars:
			t.__dict__['transforms'] = transforms_state(t.transforms, **kwargs)
	return t


def transforms_state(ts, **kwargs):
	assert isinstance(ts, collections.Sequence)

	transforms = []
	for t in ts:
		if isinstance(t, collections.Sequence):
			transforms.append(transforms_state(t, **kwargs))
		else:
			transforms.append(transform_state(t, **kwargs))
	return transforms



# Operators
'''
class Clamp(object):
	def __init__(self, min=0, max=255, soft=True, dtype='uint8'):
		self.min, self.max = min, max
		self.dtype = dtype
		self.soft = soft
		self.thresh =

	def __call__(self, img):
		if self.soft is None:
			return _clamp(img, min=self.min, max=self.max, dtype=self.dtype)
		else:
'''


class Unsqueeze(object):
	def __call__(self, img):
		if img.ndim == 2:
			return img[..., np.newaxis]
		elif img.ndim == 3:
			return img
		else:
			raise ValueError('input muse be image')



class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std 

	def __call__(self, img):
		# normalize np.ndarray or torch.FloatTensor
		if isinstance(img, np.ndarray):
			return (img - self.mean) / self.std
		elif isinstance(img, torch.FloatTensor):
			tensor = img
			for t, m, s in zip(tensor, self.mean, self.std):
				t.sub_(m).div_(s)
				return tensor 
		else:
			raise Exception('invalid input type')


class SubtractMean(object):
	# TODO: pytorch tensor
	def __init__(self, mean):
		self.mean = mean 

	def __call__(self, img):
		return img.astype(np.float32) - self.mean

class DivideBy(object):
	# TODO: pytorch tensor
	def __init__(self, divisor):
		self.divisor = divisor

	def __call__(self, img):
		return img.astype(np.float32) / self.divisor


def HalfBlood(img, anchor, f1, f2):
	assert isinstance(f1, types.LambdaType) and isinstance(f2, types.LambdaType)

	if isinstance(anchor, numbers.Number):
		anchor = int(np.ceil(anchor))

	if isinstance(anchor, int) and img.ndim == 3 and 0 < anchor < img.shape[2]:
		img1, img2 = img[:,:,:anchor], img[:,:,anchor:]

		if img1.shape[2] == 1:
			img1 = img1[:, :, 0]
		if img2.shape[2] == 1:
			img2 = img2[:, :, 0]

		img1 = f1(img1)
		img2 = f2(img2)

		if img1.ndim == 2:
			img1 = img1[..., np.newaxis]
		if img2.ndim == 2:
			img2 = img2[..., np.newaxis]
		
		return np.concatenate((img1, img2), axis=2)
	elif anchor == 0:
		img = f2(img)
		if img.ndim == 2:
			img = img[..., np.newaxis]
		return img
	else:
		img = f1(img)
		if img.ndim == 2:
			img = img[..., np.newaxis]
		return img





# Photometric Transform


class RGB2BGR(object):
	def __call__(self, img):
		assert img.ndim == 3 and img.shape[2] == 3
		return img[:, :, ::-1]

class BGR2RGB(object):
	def __call__(self, img):
		assert img.ndim == 3 and img.shape[2] == 3
		return img[:, :, ::-1]


class GrayScale(object):
	# RGB to Gray
	def __call__(self, img):
		if img.ndim == 3 and img.shape[2] == 1:
			return img

		assert img.ndim == 3 and img.shape[2] == 3
		dtype = img.dtype
		gray = np.sum(img * [0.299, 0.587, 0.114], axis=2).astype(dtype)  #5x slower than cv2.cvtColor 
		
		#gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
		return gray[..., np.newaxis]


class Hue(object):
	# skimage.color.rgb2hsv/hsv2rgb is almost 100x slower than cv2.cvtColor
	def __init__(self, var=0.05, prob=0.5, random_state=np.random):
		self.var = var
		self.prob = prob
		self.random = random_state

	def __call__(self, img):
		assert img.ndim == 3 and img.shape[2] == 3

		if self.random.random_sample() >= self.prob:
			return img

		var = self.random.uniform(-self.var, self.var)

		to_HSV, from_HSV = [(cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
							(cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR)][self.random.randint(2)]

		hsv = cv2.cvtColor(img, to_HSV).astype(np.float32)

		hue = hsv[:, :, 0] / 179. + var
		hue = hue - np.floor(hue)
		hsv[:, :, 0] = hue * 179.

		img = cv2.cvtColor(hsv.astype('uint8'), from_HSV)
		return img


class Saturation(object):
	def __init__(self, var=0.3, prob=0.5, random_state=np.random):
		self.var = var 
		self.prob = prob
		self.random = random_state

		self.grayscale = GrayScale()

	def __call__(self, img):
		if self.random.random_sample() >= self.prob:
			return img

		dtype = img.dtype
		gs = self.grayscale(img)

		alpha = 1.0 + self.random.uniform(-self.var, self.var)
		img = alpha * img.astype(np.float32) + (1 - alpha) * gs.astype(np.float32)
		return _clamp(img, dtype=dtype)



class Brightness(object):
	def __init__(self, delta=32, prob=0.5, random_state=np.random):
		self.delta = delta
		self.prob = prob
		self.random = random_state

	def __call__(self, img):
		if self.random.random_sample() >= self.prob:
			return img

		dtype = img.dtype
		#alpha = 1.0 + self.random.uniform(-self.var, self.var)
		#img = alpha * img.astype(np.float32)
		img = img.astype(np.float32) + self.random.uniform(-self.delta, self.delta)
		return _clamp(img, dtype=dtype)



class Contrast(object):
	def __init__(self, var=0.3, prob=0.5, random_state=np.random):
		self.var = var 
		self.prob = prob
		self.random = random_state

		self.grayscale = GrayScale()

	def __call__(self, img):
		if self.random.random_sample() >= self.prob:
			return img

		dtype = img.dtype
		gs = self.grayscale(img).mean()

		alpha = 1.0 + self.random.uniform(-self.var, self.var)
		img = alpha * img.astype(np.float32) + (1 - alpha) * gs
		return _clamp(img, dtype=dtype)


class RandomOrder(object):
	def __init__(self, transforms, random_state=None):  #, **kwargs):
		if random_state is None:
			self.random = np.random
		else:
			self.random = random_state
			#kwargs['random_state'] = random_state

		self.transforms = transforms_state(transforms, random=random_state)

	def __call__(self, img):
		if self.transforms is None:
			return img
		order = self.random.permutation(len(self.transforms))
		for i in order:
			img = self.transforms[i](img)
		return img


class ColorJitter(RandomOrder):
	def __init__(self, brightness=32, contrast=0.5, saturation=0.5, hue=0.1,
				 	prob=0.5, random_state=np.random):
		self.transforms = []
		self.random = random_state

		if brightness != 0:
			self.transforms.append(
				Brightness(brightness, prob=prob, random_state=random_state))
		if contrast != 0:
			self.transforms.append(
				Contrast(contrast, prob=prob, random_state=random_state))
		if saturation != 0:
			self.transforms.append(
				Saturation(saturation, prob=prob, random_state=random_state))
		if hue != 0:
			self.transforms.append(
				Hue(hue, prob=prob, random_state=random_state))



# "ImageNet Classification with Deep Convolutional Neural Networks"
# looks inferior to ColorJitter
class FancyPCA(object):
	def __init__(self, var=0.2, random_state=np.random):
		self.var = var
		self.random = random_state

		self.pca = None    # shape (channels, channels)

	def __call__(self, img):
		dtype = img.dtype
		channels = img.shape[2]
		alpha = self.random.randn(channels) * self.var

		if self.pca is None:
			pca = self._pca(img)
		else:
			pca = self.pca

		img = img + (pca * alpha).sum(axis=1)
		return _clamp(img, dtype=dtype)

	def _pca(self, img):   # single image (hwc), or a batch (nhwc)
		assert img.ndim >= 3
		channels = img.shape[-1]
		X = img.reshape(-1, channels)

		cov = np.cov(X.T)   
		evals, evecs = np.linalg.eigh(cov)
		pca = np.sqrt(evals) * evecs
		return pca

	def fit(self, imgs):   # training
		self.pca = self._pca(imgs)
		print(self.pca)


class ShuffleChannels(object):
	def __init__(self, prob=1., random_state=np.random):
		self.prob = prob
		self.random = random_state 

	def __call__(self, img):
		if self.prob < 1 and self.random.random_sample() >= self.prob:
			return img 

		assert img.ndim == 3
		permut = self.random.permutation(img.shape[2])
		img = img[:, :, permut]

		return img


# "Improved Regularization of Convolutional Neural Networks with Cutout". (arXiv:1708.04552)
# fill with 0(if image is normalized) or dataset's per-channel mean.
class Cutout(object):
	def __init__(self, size, fillval=0, prob=0.5, random_state=np.random): 
		if isinstance(size, numbers.Number):
			size = (int(size), int(size))
		self.size = size
		
		self.fillval = fillval
		self.prob = prob
		self.random = random_state

	def __call__(self, img):
		if self.random.random_sample() >= self.prob:
			return img

		h, w = img.shape[:2]
		tw, th = self.size 

		cx = self.random.randint(0, w)
		cy = self.random.randint(0, h)

		x1 = int(np.clip(cx -       tw / 2, 0, w - 1))
		x2 = int(np.clip(cx + (tw + 1) / 2, 0, w    ))
		y1 = int(np.clip(cy -       th / 2, 0, h - 1))
		y2 = int(np.clip(cy + (th + 1) / 2, 0, h    ))

		img[y1:y2, x1:x2] = self.fillval

		return img


# "Random Erasing Data Augmentation". (arXiv:1708.04896).  fill with random value
class RandomErasing(object):
	def __init__(self, area_range=(0.02, 0.2), ratio_range=[0.3, 1/0.3], fillval=None, 
				 prob=0.5, num=1, anchor=None, random_state=np.random):
		self.area_range = area_range
		self.ratio_range = ratio_range
		self.fillval = fillval
		self.prob = prob
		self.num = num
		self.anchor = anchor
		self.random = random_state

	def __call__(self, img):
		if self.random.random_sample() >= self.prob:
			return img

		h, w = img.shape[:2]

		num = self.random.randint(self.num) + 1
		count = 0
		for _ in range(10):
			area = h * w 
			target_area = _loguniform(self.area_range, self.random) * area
			aspect_ratio = _loguniform(self.ratio_range, self.random)

			tw = int(round(np.sqrt(target_area * aspect_ratio)))
			th = int(round(np.sqrt(target_area / aspect_ratio)))

			if tw <= w and th <= h:

				x1 = self.random.randint(0, w - tw + 1)
				y1 = self.random.randint(0, h - th + 1)

				fillval = self.random.randint(0, 256) if self.fillval is None else self.fillval

				erase = lambda im: self._fill(im, (x1, y1, x1+tw, y1+th), fillval)
				cut = lambda im: self._fill(im, (x1, y1, x1+tw, y1+th), 0)
				img = HalfBlood(img, self.anchor, erase, cut)

				count += 1
			if count >= num:
				return img

		# Fallback
		return img

	def _fill(self, img, rect, val):
		l, t, r, b = rect
		img[t:b, l:r] = val
		return img


#GaussianBlur
#MotionBlue
#RadialBlur
#ResizeBlur 
#Sharpen




# Geometric Transform

def _expand(img, size, lt, val):
	h, w = img.shape[:2]
	nw, nh = size 
	x1, y1 = lt 
	expand = np.zeros([nh, nw] + list(img.shape[2:]), dtype=img.dtype)
	expand[...] = val
	expand[y1: h + y1, x1: w + x1] = img
	#expand = cv2.copyMakeBorder(img, y1, nh-h-y1, x1, nw-w-x1, 
	#							cv2.BORDER_CONSTANT, value=val)  # slightly faster
	return expand


class Pad(object):
	def __init__(self, padding, fillval=0, anchor=None):
		if isinstance(padding, numbers.Number):
			padding = (padding, padding)
		assert len(padding) == 2

		self.padding = [int(np.clip(_), 0, None) for _ in padding]
		self.fillval = fillval
		self.anchor = anchor

	def __call__(self, img, cds=None):
		if max(self.padding) == 0:
			return img if cds is None else (img, cds)

		h, w = img.shape[:2]
		pw, ph = self.padding

		pad = lambda im: _expand(im, (w + pw*2, h + ph*2), (pw, ph), self.fillval)
		purer = lambda im: _expand(im, (w + pw*2, h + ph*2), (pw, ph), 0)  
		img = HalfBlood(img, self.anchor, pad, purer)

		if cds is not None:
			return img, np.array([[x + pw, y + ph] for x, y in cds])
		else:
			return img


# "SSD: Single Shot MultiBox Detector".  generate multi-resolution image/ multi-scale objects
class Expand(object):
	def __init__(self, scale_range=(1, 4), fillval=0, prob=1.0, anchor=None, random_state=np.random):
		if isinstance(scale_range, numbers.Number):
			scale_range = (1, scale_range)
		assert max(scale_range) <= 5 

		self.scale_range = scale_range	
		self.fillval = fillval
		self.prob = prob
		self.anchor = anchor
		self.random = random_state

	def __call__(self, img, cds=None):
		if self.prob < 1 and self.random.random_sample() >= self.prob:
			return img if cds is None else (img, cds)

		#multiple = _loguniform(self.scale_range, self.random)
		multiple = self.random.uniform(*self.scale_range)

		h, w = img.shape[:2]
		nh, nw = int(multiple * h), int(multiple * w)

		if multiple < 1:
			return RandomCrop(size=(nw, nh), random_state=self.random)(img, cds)

		y1 = self.random.randint(0, nh - h + 1)
		x1 = self.random.randint(0, nw - w + 1)

		expand = lambda im: _expand(im, (nw, nh), (x1, y1), self.fillval)
		purer = lambda im: _expand(im, (nw, nh), (x1, y1), 0)
		img = HalfBlood(img, self.anchor, expand, purer)

		if cds is not None:
			return img, np.array([[x + x1, y + y1] for x, y in cds])
		else:
			return img


# scales the smaller edge to given size
class Scale(object):
	def __init__(self, size, mode='linear', lazy=False, anchor=None, random_state=np.random):
		assert isinstance(size, int)

		self.size = int(size)
		self.mode = mode
		self.lazy = lazy
		self.anchor = anchor
		self.random = random_state

	def __call__(self, img, cds=None):
		interp_mode = (self.random.choice(list(InterpolationFlags.values())) if self.mode is None 
					   			else InterpolationFlags.get(self.mode, cv2.INTER_LINEAR))

		h, w = img.shape[:2]

		if self.lazy and min(h, w) >= self.size:
			return img if cds is None else (img, cds)

		if h < w:
			tw, th = int(self.size / float(h) * w), self.size
		else:
			th, tw = int(self.size / float(w) * h), self.size

		# skimage.transform.resize 10x slower than cv2.resize
		resize = lambda im: cv2.resize(im, (tw, th), interpolation=interp_mode)
		purer = lambda im: cv2.resize(im, (tw, th), interpolation=cv2.INTER_NEAREST)
		img = HalfBlood(img, self.anchor, resize, purer)

		if cds is not None:
			s_x, s_y = tw / float(w), th / float(h)
			return img, np.array([[x * s_x, y * s_y] for x, y in cds])
		else:
			return img


class RandomScale(object):
	def __init__(self, size_range, mode='linear', anchor=None, random_state=np.random):
		assert isinstance(size_range, collections.Sequence) and len(size_range) == 2

		self.size_range = size_range
		self.mode = mode
		self.anchor = anchor
		self.random = random_state

	def __call__(self, img, cds=None):
		interp_mode = (self.random.choice(list(InterpolationFlags.values())) if self.mode is None 
					   			else InterpolationFlags.get(self.mode, cv2.INTER_LINEAR))

		h, w = img.shape[:2]
		size = int(self.random.uniform(*self.size_range))
		
		if h < w:
			tw, th = int(size / float(h) * w), size
		else:
			th, tw = int(size / float(w) * h), size

		resize = lambda im: cv2.resize(im, (tw, th), interpolation=interp_mode)
		purer = lambda im: cv2.resize(im, (tw, th), interpolation=cv2.INTER_NEAREST)
		img = HalfBlood(img, self.anchor, resize, purer)

		if cds is not None:
			s_x, s_y = tw / float(w), th / float(h)
			return img, np.array([[x * s_x, y * s_y] for x, y in cds])
		else:
			return img


class CenterCrop(object):
	def __init__(self, size):
		if isinstance(size, numbers.Number):
			size = (int(size), int(size))
		self.size = size

	def __call__(self, img, cds=None):
		h, w = img.shape[:2]
		tw, th = self.size

		if h == th and w == tw:
			return img if cds is None else (img, cds)
		elif h < th or w < tw:
			raise Exception('invalid crop size')

		x1 = int(round((w - tw) / 2.))
		y1 = int(round((h - th) / 2.))
		img = img[y1:y1 + th, x1:x1 + tw]

		if cds is not None:
			return img, _coords_clamp([[x - x1, y - y1] for x, y in cds], img.shape)
		else:
			return img


class RandomCrop(object):
	def __init__(self, size, fillval=0, random_state=np.random):
		if isinstance(size, numbers.Number):
			size = (int(size), int(size))
		self.size = size
		self.random = random_state

	def __call__(self, img, cds=None):
		h, w = img.shape[:2]
		tw, th = self.size

		assert h >= th and w >= tw

		x1 = self.random.randint(0, w - tw + 1)
		y1 = self.random.randint(0, h - th + 1)
		img = img[y1:y1 + th, x1:x1 + tw]

		if cds is not None:
			return img, _coords_clamp([[x - x1, y - y1] for x, y in cds], img.shape)
		else:
			return img

'''
# "SSD: Single Shot MultiBox Detector". 
# object-aware RandomCrop, crop multi-scale objects
class ObjectRandomCrop(object):
	def __init__(self, final_size=None, prob=1., random_state=np.random):
		self.final_size = final_size   # reference size
		self.final_area = (final_size * final_size if isinstance(final_size, numbers.Number) 
									else np.prod(final_size))
		self.prob = prob
		self.random = random_state 

		self.options = [
			None,                # keep original size
			#(-np.inf, 0.1),      # large crop
			(0.02, 0.1),
			(0.1, 0.3),
			(0.3, 0.5),
			(0.5, 0.7),
			(0.7, np.inf),       # small crop
			(-np.inf, np.inf),   # arbitrary size  
		]

	def __call__(self, img, cbs):
		h, w = img.shape[:2]

		# ad-hoc
		if len(cbs) == 0:
			return img, cbs

		if len(cbs[0]) == 4:
			boxes = cbs
		elif len(cbs[0]) == 2:
			boxes = _to_bboxes(cbs, img.shape[:2])
		else:
			raise Exception('invalid input')

		for attempt in range(30):
			mode = self.random.choice(self.options)

			if mode is None or (self.prob < 1 and self.random.random_sample() >= self.prob):
				if self.final_size is not None:
					# area constraint
					box_areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
					size = np.sqrt(box_areas * self.final_area / (h * w))
					if not ((11 < size) * (size < 18)).any() and size.max() > 22:
						return img, cbs
					mode = self.options[self.random.randint(1, len(self.options))]
				else:
					return img, cbs

			min_iou, max_iou = mode

			for _ in range(50):
				tw = self.random.uniform(0.3 * w, w)
				th = self.random.uniform(0.3 * h, h)
				if max(th / tw, tw / th) > 2:
					continue

				x1 = self.random.randint(0, w - tw + 1)
				y1 = self.random.randint(0, h - th + 1)

				rect = np.array([int(x1), int(y1), int(x1+tw), int(y1+th)])
				jaccard, coverage, inter = _jaccard(boxes, rect)

				# iou constraint
				if jaccard.max() < min_iou or jaccard.max() > max_iou:
					continue

				# coverage constraint
				m1 = coverage > 1/9.
				m2 = coverage < 0.45
				if (m1 * m2).any():
					continue

				mask = coverage >= 0.45
				if not mask.any():
					continue

				# area constraint
				if self.final_size is not None:
					area = (rect[2] - rect[0]) * (rect[3] - rect[1]) * 1.
					size = np.sqrt(inter[mask] * self.final_area / area)
					if ((11 < size) * (size < 18)).any() or size.max() < 22:
						continue

				img = img[rect[1]:rect[3], rect[0]:rect[2]]

				boxes[:, :2] = np.clip(boxes[:, :2], rect[:2], rect[2:])
				boxes[:, :2] = boxes[:, :2] - rect[:2]
				boxes[:, 2:] = np.clip(boxes[:, 2:], rect[:2], rect[2:])
				boxes[:, 2:] = boxes[:, 2:] - rect[:2]
				boxes[np.logical_not(mask), :] = 0

				#print(min_iou, max_iou)

				if len(cbs[0]) == 4:
					return img, boxes
				else:
					return img, _to_coords(boxes)

		# Fallback
		return img, cbs
'''

class ObjectRandomCrop(object):
	def __init__(self, prob=1., random_state=np.random):
		self.prob = prob
		self.random = random_state 

		self.options = [
		#(0, None), 
		(0.1, None),     
		(0.3, None),
		(0.5, None),
		(0.7, None),
		(0.9, None),       
		(None, 1), ]
	

	def __call__(self, img, cbs):
		h, w = img.shape[:2]

		if len(cbs) == 0:
			return img, cbs

		# ad-hoc
		if len(cbs[0]) == 4:
			boxes = cbs
		elif len(cbs[0]) == 2:
			boxes = _to_bboxes(cbs, img.shape[:2])
		else:
			raise Exception('invalid input')

		params = [(np.array([0, 0, w, h]), None)]

		for min_iou, max_iou in self.options:
			if min_iou is None:
				min_iou = 0
			if max_iou is None:
				max_iou = 1

			for _ in range(50):
				scale = self.random.uniform(0.3, 1)
				aspect_ratio = self.random.uniform(
					max(1 / 2., scale * scale),
					min(2., 1 / (scale * scale)))
				th = int(h * scale / np.sqrt(aspect_ratio))
				tw = int(w * scale * np.sqrt(aspect_ratio))

				x1 = self.random.randint(0, w - tw + 1)
				y1 = self.random.randint(0, h - th + 1)
				rect = np.array([x1, y1, x1 + tw, y1 + th])

				iou, coverage, _ = _jaccard(boxes, rect)

				#m1 = coverage > 0.1
				#m2 = coverage < 0.45
				#if (m1 * m2).any():
				#	continue

				center = (boxes[:, :2] + boxes[:, 2:]) / 2
				mask = np.logical_and(rect[:2] <= center, center < rect[2:]).all(axis=1)

				#mask = coverage >= 0.45
				#mask
				if not mask.any():
					continue

				if min_iou <= iou.max() and iou.min() <= max_iou:
					params.append((rect, mask))
					break
		rect, mask = params[self.random.randint(len(params))]

		img = img[rect[1]:rect[3], rect[0]:rect[2]]
		boxes[:, :2] = np.clip(boxes[:, :2], rect[:2], rect[2:])
		boxes[:, :2] = boxes[:, :2] - rect[:2]
		boxes[:, 2:] = np.clip(boxes[:, 2:], rect[:2], rect[2:])
		boxes[:, 2:] = boxes[:, 2:] - rect[:2]
		if mask is not None:
			boxes[np.logical_not(mask), :] = 0

		if len(cbs[0]) == 4:
			return img, boxes
		else:
			return img, _to_coords(boxes)





# Random crop with size 8%-100% and aspect ratio 3/4 - 4/3. (Inception-style)
class RandomSizedCrop(object):
	def __init__(self, size, mode='linear', anchor=None, random_state=np.random):
		self.size = size 
		self.mode = mode
		self.anchor = anchor
		self.random = random_state

		self.scale = Scale(size, mode=mode, anchor=anchor)
		self.crop = CenterCrop(size)

	def __call__(self, img, cds=None):
		interp_mode = (self.random.choice(list(InterpolationFlags.values())) if self.mode is None 
					   			else InterpolationFlags.get(self.mode, cv2.INTER_LINEAR))

		h, w = img.shape[:2]

		for _ in range(10):
			area = h * w
			target_area = self.random.uniform(0.16, 1.0) * area   # 0.08~1.0
			aspect_ratio = self.random.uniform(3. / 4, 4. / 3)

			tw = int(round(np.sqrt(target_area * aspect_ratio)))
			th = int(round(np.sqrt(target_area / aspect_ratio)))

			if self.random.random_sample() < 0.5:
				tw, th = th, tw 

			if tw <= w and th <= h:
				x1 = self.random.randint(0, w - tw + 1)
				y1 = self.random.randint(0, h - th + 1)

				img = img[y1:y1 + th, x1:x1 + tw]

				resize = lambda im: cv2.resize(im, (self.size, self.size), interpolation=interp_mode)
				purer = lambda im: cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
				img = HalfBlood(img, self.anchor, resize, purer)

				if cds is not None:
					scale_x = self.size / float(tw)
					scale_y = self.size / float(th)

					return img, _coords_clamp([[scale_x*(x-x1), scale_y*(y-y1)] for x, y in cds], img.shape)
				else:
					return img

		# Fallback
		return self.crop(self.scale(img, cds=cds), cds=cds)


class GridCrop(object):
	def __init__(self, size, grid=5, random_state=np.random):
		# 4 grids, 5 grids or 9 grids
		if isinstance(size, numbers.Number):
			size = (int(size), int(size))
		self.size = size

		self.grid = grid
		self.random = random_state
		self.map = {
			0: lambda w, h, tw, th: (            0,            0),
			1: lambda w, h, tw, th: (       w - tw,            0),
			2: lambda w, h, tw, th: (       w - tw,       h - th),
			3: lambda w, h, tw, th: (            0,       h - th),
			4: lambda w, h, tw, th: ((w - tw) // 2, (h - th) // 2),
			5: lambda w, h, tw, th: ((w - tw) // 2,            0),
			6: lambda w, h, tw, th: (       w - tw, (h - th) // 2),
			7: lambda w, h, tw, th: ((w - tw) // 2,       h - th),
			8: lambda w, h, tw, th: (            0, (h - th) // 2),
		}

	def __call__(self, img, cds=None, index=None):
		h, w = img.shape[:2]
		tw, th = self.size
		if index is None:
			index = self.random.randint(0, self.grid)
		if index not in self.map:
			raise Exception('invalid index')

		x1, y1 = self.map[index](w, h, tw, th)
		img = img[y1:y1 + th, x1:x1 + tw]

		if cds is not None:
			return img, _coords_clamp([[x - x1, y - y1] for x, y in cds], img.shape)
		else:
			return img



class Resize(object):
	def __init__(self, size, mode='linear', anchor=None, random_state=np.random):
		if isinstance(size, numbers.Number):
			size = (int(size), int(size))
		self.size = size

		self.mode = mode
		self.anchor = anchor
		self.random = random_state

	def __call__(self, img, cds=None):
		interp_mode = (self.random.choice(list(InterpolationFlags.values())) if self.mode is None 
					   			else InterpolationFlags.get(self.mode, cv2.INTER_LINEAR))

		h, w = img.shape[:2]
		tw, th = self.size

		resize = lambda im: cv2.resize(im, (tw, th), interpolation=interp_mode)
		purer = lambda im: cv2.resize(im, (tw, th), interpolation=cv2.INTER_NEAREST)
		img = HalfBlood(img, self.anchor, resize, purer)

		if cds is not None:
			s_x = tw / float(w)
			s_y = th / float(h)
			return img, np.array([[s_x * x, s_y * y] for x, y in cds])
		else:
			return img


class RandomResize(object):
	def __init__(self, scale_range=(0.8, 1.2), ratio_range=1., mode='linear', anchor=None,
				 random_state=np.random):

		sr = scale_range
		if isinstance(sr, numbers.Number):
			sr = (min(sr, 1. / sr), max(sr, 1. / sr))
		assert  max(sr) <= 5
		self.sr = sr

		rr = ratio_range
		if isinstance(rr, numbers.Number):
			rr = (min(rr, 1. / rr), max(rr, 1. / rr))
		assert  max(rr) <= 5
		self.rr = rr
		
		self.mode = mode
		self.anchor = anchor
		self.random = random_state

	def __call__(self, img, cds=None):
		interp_mode = (self.random.choice(list(InterpolationFlags.values())) if self.mode is None 
					   			else InterpolationFlags.get(self.mode, cv2.INTER_LINEAR))

		h, w = img.shape[:2]

		scale_factor = _loguniform(self.sr, self.random)
		ratio_factor = _loguniform(self.rr, self.random)

		th = int(h * scale_factor)
		tw = int(w * scale_factor * ratio_factor)

		resize = lambda im: cv2.resize(im, (tw, th), interpolation=interp_mode)
		purer = lambda im: cv2.resize(im, (tw, th), interpolation=cv2.INTER_NEAREST)
		img = HalfBlood(img, self.anchor, resize, purer)
		
		if cds is not None:
			s_x = tw / float(w)
			s_y = th / float(h)
			return img, np.array([[s_x * x, s_y * y] for x, y in cds])
		else:
			return img


class ElasticTransform(object):
	def __init__(self, alpha=1000, sigma=40, mode='linear', border='constant', fillval=0, 
				 anchor=None, random_state=np.random):

		if isinstance(fillval, numbers.Number):
			fillval = [fillval] * 3

		self.alpha, self.sigma = alpha, sigma
		self.mode = mode
		self.border = BorderTypes.get(border, cv2.BORDER_REPLICATE)
		self.fillval = fillval
		self.anchor = anchor
		self.random = random_state


	def __call__(self, img, cds=None):
		interp_mode = (self.random.choice(list(InterpolationFlags.values())) if self.mode is None 
					   			else InterpolationFlags.get(self.mode, cv2.INTER_LINEAR))

		shape = img.shape[:2]

		ksize = self.sigma * 4 + 1
		dx = cv2.GaussianBlur((self.random.rand(*img.shape[:2]) * 2 - 1).astype(np.float32), 
							  (ksize, ksize), 0) * self.alpha
		dy = cv2.GaussianBlur((self.random.rand(*img.shape[:2]) * 2 - 1).astype(np.float32), 
							  (ksize, ksize), 0) * self.alpha

		y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
		mapy, mapx = (y + dy).astype(np.float32), (x + dx).astype(np.float32)

		elastic = lambda im: cv2.remap(im, mapx, mapy, interpolation=interp_mode, borderMode=self.border, borderValue=self.fillval)
		purer = lambda im: cv2.remap(im, mapx, mapy, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
		img = HalfBlood(img, self.anchor, elastic, purer)

		if cds is None:
			return img
		else:
			cds_from = np.hstack([mapx.reshape(-1, 1), mapy.reshape(-1, 1)])
			cds_to = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
			cds_ = []
			for coord in cds:
				# TODO: top-k
				ind = np.argmin(np.sum((coord - cds_from)**2, axis=1))
				cds_.append(cds_to[ind])
			return img, _coords_clamp(cds_, img.shape)


class RandomRotate(object):
	def __init__(self, angle_range=(-30.0, 30.0), mode='linear', border='constant', fillval=0, 
				 anchor=None, random_state=np.random):   
		if isinstance(angle_range, numbers.Number):
			angle_range = (-angle_range, angle_range)
		self.angle_range = angle_range

		if isinstance(fillval, numbers.Number):
			fillval = [fillval] * 3

		self.mode = mode
		self.border = BorderTypes.get(border, cv2.BORDER_REPLICATE)
		self.fillval = fillval
		self.anchor = anchor
		self.random = random_state

	def __call__(self, img, cds=None):
		interp_mode = (self.random.choice(list(InterpolationFlags.values())) if self.mode is None 
					   			else InterpolationFlags.get(self.mode, cv2.INTER_LINEAR))

		h, w = img.shape[:2]
		angle = self.random.uniform(*self.angle_range)

		M = cv2.getRotationMatrix2D((w/2., h/2.), angle, 1)

		rotate = lambda im: cv2.warpAffine(im, M, dsize=(w, h), flags=self.mode, borderMode=self.border, borderValue=self.fillval)
		purer = lambda im: cv2.warpAffine(im, M, dsize=(w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
		img = HalfBlood(img, self.anchor, rotate, purer)

		if cds is not None:
			cos = np.cos(angle * np.pi / 180.)
			sin = np.sin(angle * np.pi / 180.)
			cds_ = []
			for x, y in cds:
				x, y = x - w/2., -(y - h/2.)
				x, y = cos*x - sin*y, sin*x + cos*y
				x, y = x + w/2., -y + h/2.
				cds_.append([x, y])
			return img, _coords_clamp(cds_, img.shape)
		else:
			return img


class Rotate90(object):
	def __init__(self, random_state=np.random):
		# 4 directions
		self.random = random_state

		self.map = {
			0: lambda x, y, w, h: (    x,     y),
			1: lambda x, y, w, h: (    y, w-1-x),
			2: lambda x, y, w, h: (w-1-x, h-1-y),
			3: lambda x, y, w, h: (h-1-y,     x),
		}

	def __call__(self, img, cds=None, index=None):
		h, w = img.shape[:2]
		if index is None:
			index = self.random.randint(0, 4)
		if index not in self.map:
			raise Exception('invalid index')

		img = np.rot90(img, index)

		if cds is not None:
			return img, np.array([self.map[index](x, y, w, h) for x, y in cds])
		else:
			return img


class RandomShift(object):
	def __init__(self, tx=(-0.1, 0.1), ty=None, border='constant', fillval=0, anchor=None, random_state=np.random):   
		if isinstance(tx, numbers.Number):
			tx = (-abs(tx), abs(tx))
		assert isinstance(tx, tuple) and np.abs(tx).max() < 1
		if ty is None:
			ty = tx
		elif isinstance(ty, numbers.Number):
			ty = (-abs(ty), abs(ty))
		assert isinstance(ty, tuple) and np.abs(ty).max() < 1
		self.tx, self.ty = tx, ty

		if isinstance(fillval, numbers.Number):
			fillval = [fillval] * 3

		self.border = BorderTypes.get(border, cv2.BORDER_REPLICATE)
		self.fillval = fillval
		self.anchor = anchor
		self.random = random_state

	def __call__(self, img, cds=None):
		h, w = img.shape[:2]
		tx = self.random.uniform(*self.tx) * w 
		ty = self.random.uniform(*self.ty) * h

		M = np.float32([[1,0,tx],[0,1,ty]])

		shift = lambda im: cv2.warpAffine(im, M, dsize=(w, h), flags=cv2.INTER_NEAREST, borderMode=self.border, borderValue=self.fillval)
		purer = lambda im: cv2.warpAffine(im, M, dsize=(w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
		img = HalfBlood(img, self.anchor, shift, purer)

		if cds is not None:
			return img, _coords_clamp([[x + tx, y + ty] for x, y in cds], img.shape)
		else:
			return img


class HorizontalFlip(object):
	def __init__(self, prob=0.5, random_state=np.random):
		self.prob = prob
		self.random = random_state

	def __call__(self, img, cds=None, flip=None):
		if flip is None:
			flip = self.random.random_sample() < self.prob

		if flip:
			img = img[:, ::-1]
		
		if cds is not None:
			h, w = img.shape[:2]
			t = lambda x, y: [w-1-x, y] if flip else [x, y]
			return img, np.array([t(x, y) for x, y in cds])
		else:
			return img


class VerticalFlip(object):
	def __init__(self, prob=0.5, random_state=np.random):
		self.prob = prob
		self.random = random_state

	def __call__(self, img, cds=None, flip=None):
		if flip is None:
			flip = self.random.random_sample() < self.prob

		if flip:
			img = img[::-1, :]
		
		if cds is not None:
			h, w = img.shape[:2]
			t = lambda x, y: [x, h-1-y] if flip else [x, y]
			return img, np.array([t(x, y) for x, y in cds])
		else:
			return img


'''class RandomFlip(object):
	def __init__(self, random_state=np.random):
		self.random = random_state
		self.transforms = [HorizontalFlip(random_state=np.random), 
						   VerticalFlip(random_state=np.random)]

	def __call__(self, img, cds=None, index=None):
		hori, vert = self.transforms
		if index is None:
			index = self.random.randint(0, 4)

		if index == 0:
			return hori(img, cds=cds, flip=False)
		elif index == 1:
			return hori(img, cds=cds, flip=True)
		elif index == 2:
			return vert(img, cds=cds, flip=True)
		elif index == 3:
			if cds is None:
				img = hori(img, flip=True)
				return vert(img, flip=True)
			else:
				img, cds = hori(img, cds=cds, flip=True)
				return vert(img, cds=cds, flip=False)
		else:
			raise Exception('invalid index')'''




# Pipeline

class Lambda(object):
	def __init__(self, lambd):
		assert isinstance(lambd, types.LambdaType)
		self.lambd = lambd 

	def __call__(self, *args):
		return self.lambd(*args)


class Merge(object):
	def __init__(self, axis=-1):
		self.axis = axis

	def __call__(self, *imgs):
		# ad-hoc 
		if len(imgs) > 1 and not isinstance(imgs[0], collections.Sequence):
			pass
		elif len(imgs) == 1 and isinstance(imgs[0], collections.Sequence):   # unreliable
			imgs = imgs[0]
		elif len(imgs) == 1:
			return imgs[0]
		else:
			raise Exception('input must be a sequence (list, tuple, etc.)')

		assert len(imgs) > 0 and all([isinstance(_, np.ndarray)
					for _ in imgs]), 'only support numpy array'

		shapes = []
		imgs_ = []
		for i, img in enumerate(imgs):
			if img.ndim == 2:
				img = np.expand_dims(img, axis=self.axis)
			imgs_.append(img)
			shape = list(img.shape)
			shape[self.axis] = None
			shapes.append(shape)
		assert all([_ == shapes[0] for _ in shapes]), 'shapes must match'
		return np.concatenate(imgs_, axis=self.axis)


class Split(object):
	def __init__(self, *slices, **kwargs):
		slices_ = []
		for s in slices:
			if isinstance(s, collections.Sequence):
				slices_.append(slice(*s))
			else:
				slices_.append(s)
			assert all([isinstance(s, slice) for s in slices_]), 'slices must consist of slice instances'

		self.slices = slices_
		self.axis = kwargs.get('axis', -1)

	def __call__(self, img):
		if isinstance(img, np.ndarray):
			result = []
			for s in self.slices:
				sl = [slice(None)] * img.ndim 
				sl[self.axis] = s 
				result.append(img[sl])
			return result
		else:
			raise Exception('object must be a numpy array')


class Branching(object):
	# TODO
	pass

class Bracket(object):
	# TODO
	pass

class Flatten(object):
	# TODO 
	pass

class Permute(object):
	# TODO
	pass


class Compose(object):
	def __init__(self, transforms, random_state=None, **kwargs):
		if random_state is not None:
			kwargs['random_state'] = random_state
		self.transforms = transforms_state(transforms, **kwargs)

	def __call__(self, *data):
		# ad-hoc 
		if len(data) >= 1 and not isinstance(data[0], collections.Sequence):
			pass
		elif len(data) == 1 and isinstance(data[0], collections.Sequence) and len(data[0]) > 0:   # unreliable
			data = list(data[0])
		else:
			raise Exception('invalid input')

		for t in self.transforms:
			if not isinstance(data, collections.Sequence):   # unreliable
				data = [data]

			if isinstance(t, collections.Sequence):
				if len(t) > 1:
					assert isinstance(data, collections.Sequence) and len(data) == len(t)
					ds = []
					for i, d in enumerate(data):
						if callable(t[i]):
							ds.append(t[i](d))
						else:
							ds.append(d)
					data = ds
				elif len(t) == 1:
					if callable(t[0]):
						data = [t[0](data[0])] + list(data)[1:]
			elif callable(t):
				data = t(*data)
			elif t is not None:
				raise Exception('invalid transform type')

		if isinstance(data, collections.Sequence) and len(data) == 1:   # unreliable
			return data[0]
		else:
			return data

	def set_random_state(self, random_state):
		self.transforms = transforms_state(self.transforms, random=random_state)


class RandomCompose(Compose):
	def __init__(self, transforms, random_state=None, **kwargs):
		if random_state is None:
			random_state = np.random
		else:
			kwargs['random_state'] = random_state

		self.transforms = transforms_state(transforms, **kwargs)
		self.random = random_state

	def __call__(self, *data):
		self.random.shuffle(self.transforms)

		return super(RandomCompose, self).__call__(*data)


class ToNumpy(object):
	def __call__(self, pic):
		# torch.FloatTensor -> np.ndarray
		# or PIL Image -> np.ndarray
		# TODO
		pass


class ToTensor(object):
	def __init__(self):
		pass

	def __call__(self, img):
		# np.ndarray -> torch.FloatTensor
		# or PIL Image -> torch.FloatTensor

		# TODO: add option to choose whether div(255)
		if isinstance(img, np.ndarray):
			img = img.transpose((2, 0, 1))
			return torch.from_numpy(np.ascontiguousarray(img)).float()
		else:
			# TODO
			pass

class ToLongTensor(object):
	def __init__(self):
		pass 
	
	def __call_(self, label):
		if isinstance(label, np.ndarray):
			return torch.from_numpy(label).long()


class BoxesToCoords(object):
	def __init__(self, relative=False):
		self.relative = relative

	def bbox2coords(self, bbox):
		xmin, ymin, xmax, ymax = bbox 
		return np.array([
			[xmin, ymin],
			[xmax, ymin],
			[xmax, ymax],
			[xmin, ymax],
		])

	def __call__(self, img, boxes):
		if len(boxes) == 0:
			return img, np.array([])

		h, w = img.shape[:2]
		if self.relative:
			boxes[:, 0] *= w
			boxes[:, 2] *= w
			boxes[:, 1] *= h
			boxes[:, 3] *= h
		return img, np.vstack([self.bbox2coords(_) for _ in boxes])


class CoordsToBoxes(object):
	def __init__(self, relative=True):
		self.relative = relative

	def coords2bbox(self, cds, w, h):
		xmin = np.clip(cds[:, 0].min(), 0, w - 1)
		xmax = np.clip(cds[:, 0].max(), 0, w - 1)
		ymin = np.clip(cds[:, 1].min(), 0, h - 1)
		ymax = np.clip(cds[:, 1].max(), 0, h - 1)
		return np.array([xmin, ymin, xmax, ymax])

	def __call__(self, img, cds):
		if len(cds) == 0:
			return img, np.array([])

		assert len(cds) % 4 == 0
		num = len(cds) // 4

		h, w = img.shape[:2]
		boxcds = np.split(np.array(cds), np.arange(1, num) * 4)
		boxes = np.array([self.coords2bbox(_, w, h) for _ in boxcds])

		if self.relative:
			boxes[:, 0] /= float(w) 
			boxes[:, 2] /= float(w)
			boxes[:, 1] /= float(h)
			boxes[:, 3] /= float(h)

		return img, boxes


class OneHotMask(object):
	def __init__(self, n_classes):
		self.n_classes = n_classes

	def __call__(self, mask):
		if mask.ndim == 3 and mask.shape[2] == 1:
			mask = mask[:, :, 0]
		assert mask.ndim == 2 and mask.max() < self.n_classes

		onehot_mask = np.zeros((mask.shape[0], mask.shape[1], self.n_classes), dtype=np.uint8)
		for i in range(self.n_classes):
			onehot_mask[:, :, i] = mask == i
		return onehot_mask