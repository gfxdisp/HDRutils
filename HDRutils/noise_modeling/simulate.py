from abc import ABC, abstractmethod
import numpy as np, logging

logger = logging.getLogger(__name__)

class NoiseModel(ABC):
	@abstractmethod
	def simulate(self):
		pass


class PoissonNormalNoise(NoiseModel):
	"""Compund Poisson-normal noise from our HDR merging paper"""
	def __init__(self, camera_name):
		self.cam = self.from_preset(camera_name)


	def from_preset(self, camera_name):
		"""Cameras measured in Cambridge using calibration target"""
		Camera = {
				  'SonyA7r1': {'k':[0.327064, 0.330341, 0.32096], 'std_read':0.699102, 'std_adc':0.0358603, 'bits':14},
				  'CanonT1': {'k':[1.36308, 1.18256, 1.15319], 'std_read':0.938028, 'std_adc':5.00525, 'bits':14},
				  'SonyA7r3': {'k':[0.422044, 0.384277, 0.388967], 'std_read':0.705127, 'std_adc':3.02845, 'bits':14},
				  'SamsungS9': {'k':[0.303362, 0.312534, 0.320971], 'std_read':1.06331, 'std_adc':2.3726, 'bits':10},
				  'Empty': {'k':[1.,1.,1.], 'std_read':1., 'std_adc':1., 'bits':14}
		}
		cam = Camera[camera_name]
		cam['k'] = np.array(cam['k'])
		return cam


	def simulate(self, phi, exp, iso, disable_static_noise=False):
		t = float(exp)
		g = float(iso) / 100
		img = phi * t * g
		cam_max = 2**self.cam['bits'] - 1
		if img.max() > cam_max:
			logger.warning(f'Max pixel is {img.max()} before adding noise. Values > {cam_max} '
						   f'are likely to be clipped due to saturation')

		# Sample noise (Eq. 2)
		var = img * g * self.cam['k'][None,None,:]
		if not disable_static_noise:
			var += ((self.cam['std_read'] * g)**2 + (self.cam['std_adc'])**2) * self.cam['k'][None,None,:]**2
		noise = np.random.normal(scale=np.sqrt(var))

		quantized = (img + noise).clip(0, cam_max).astype(np.uint16)
		return quantized


class NormalNoise(NoiseModel):
	"""
	Normal approximation used by Dartable database
	https://www.darktable.org/2012/12/profiling-sensor-and-photon-noise/
	"""
	def __init__(self, make=None, model=None, iso=None, bits=14):
		import json, os
		with open(os.path.join(os.path.dirname(__file__), 'darktable.json')) as f:
			self.data = json.load(f)['noiseprofiles']
		self.makes = []
		for m in self.data:
			self.makes.append(m['maker'])
		logger.info('Camera database successfully loaded')
		if make and model and iso:
			self.set_profile(make, model, iso)
		else:
			logger.warning('Camera make, model and iso are not specified')
		self.bits = bits


	def set_profile(self, make_str, model_str, iso):
		"""
		Set camera parameters from camera presets

		:make_str: Pick 1 from ['Canon', 'Fujifilm', 'Minolta', 'Nikon', 'Olympus', 'Panasonic',
				   'Pentax', 'Samsung', 'Sony', 'LGE', 'Ricoh', 'Leica', 'YI TECHNOLOGY']
		:model_str: Specific models
		:iso: ISO to simulate
		"""

		iso = int(iso)
		make_id = self.makes.index(make_str)
		make = self.data[make_id]['models']
		for m in make:
			if model_str == m['model']:
				model = m
				for i in model['profiles']:
					if iso == i['iso']:
						self.a = np.array(i['a'])
						self.b = np.array(i['b'])
						self.g = iso/100
						return
		logger.error(f'Incorrect make ({make_str}), model ({model_str}) and iso ({iso})')


	def var(self, img, make=None, model=None, iso=None, disable_static_noise=False):
		if make and model and iso:
			self.set_profile(make, model, iso)

		# Darktable parameters are for normalized images
		if img.max() > 1:
			logger.warning(f'Max pixel is {img.max()} before adding noise. Values > 1 are '
						   f'likely to be clipped due to saturation')

		# Sample noise
		if img.ndim == 3:
			var = img*self.a[None,None,:]
			if not disable_static_noise:
				var += self.b[None,None,:]
		elif img.ndim == 2:
			# Assume RGGB
			# TODO: read from exif
			var = np.zeros_like(img)
			b = self.b if not disable_static_noise else np.zeros(3)
			var[::2,::2] = img[::2,::2]*self.a[0] + b[0]
			var[::2,1::2] = img[::2,1::2]*self.a[1] + b[1]
			var[1::2,::2] = img[1::2,::2]*self.a[1] + b[1]
			var[1::2,1::2] = img[1::2,1::2]*self.a[2] + b[2]
		if var.min() < 0:
			eps = np.abs(var).min()
			logger.warning('Predicted variance is non-positive for some pixels. These will be clipped.')
			var = np.maximum(var, eps)
		logger.info(f'Variance statistics: {var.min()}, {var.mean()}, {var.max()}')
		return var


	def simulate(self, phi, exp, iso=None, make=None, model=None, disable_static_noise=False, black_level=0):
		t = float(exp)

		img = phi * t * self.g
		var = self.var(img, make, model, iso, disable_static_noise)
		noise = np.random.normal(scale=np.sqrt(var))

		dtype = np.uint8 if self.bits <= 8 else np.uint16
		logger.info(f'Quantizing to type {dtype} since bit-depth is {self.bits}')
		max_value = 2**self.bits - 1
		quantized = ((img + noise + black_level/max_value).clip(0, 1)*max_value).astype(dtype)
		return quantized


	def set_bayer(self, size):
		# Assume RGGB
		assert len(size) == 2
		self.bayer_a, self.bayer_b = np.zeros([2] + list(size))
		self.bayer_a[::2,::2] = self.a[0]
		self.bayer_a[::2,1::2] = self.a[1]
		self.bayer_a[1::2,::2] = self.a[1]
		self.bayer_a[1::2,1::2] = self.a[2]

		self.bayer_b[::2,::2] = self.b[0]
		self.bayer_b[::2,1::2] = self.b[1]
		self.bayer_b[1::2,::2] = self.b[1]
		self.bayer_b[1::2,1::2] = self.b[2]
