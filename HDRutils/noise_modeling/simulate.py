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
	def __init__(self):
		import json, os
		with open(os.path.join(os.path.dirname(__file__), 'darktable.json')) as f:
			self.data = json.load(f)['noiseprofiles']
		self.makes = []
		for make in self.data:
			self.makes.append(make['maker'])
	
	def get_profile(self, make_str, model_str, iso):
		"""
		Obtain camera parameters from camera presets

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
						a = np.array(i['a'])
						b = np.array(i['b'])
						return a, b, iso/100
		logger.error(f'Incorrect make ({make_str}), model ({model_str}) and iso ({iso})')

	def simulate(self, phi, make_str, model_str, exp, iso, disable_static_noise=False, bits=8):
		assert bits <= 16
		a, b, g = self.get_profile(make_str, model_str, iso)
		t = float(exp)

		# Darktable parameters are for normalized images
		img = phi * t * g
		if img.max() > 1:
			logger.warning(f'Max pixel is {img.max()} before adding noise. Values > 1 are '
						   f'likely to be clipped due to saturation')

		# Sample noise
		var = img*a[None,None,:]
		if not disable_static_noise:
			var += b[None,None,:]
		logger.info(f'Var statistics: {var.min()}, {var.mean()}, {var.max()}')
		noise = np.random.normal(scale=np.sqrt(var))

		dtype = np.uint8 if bits <= 8 else np.uint16
		max_value = 2**bits - 1
		quantized = ((img + noise).clip(0, 1)*max_value).astype(dtype)
		return quantized
