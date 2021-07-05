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

	def simulate(self, img, iso, exp):
		# Scale image according to match camera bits
		scaling_factor = img.max() /  (2**self.cam['bits'] - 1)
		img = img / scaling_factor
		t = float(exp)
		g = float(iso) / 100

		var = (img/t + (self.cam['std_read']/t)**2 + \
			  (self.cam['std_adc'] / t / g)**2) * self.cam['k'][None,None,:]**2
		noise = np.random.normal(scale=np.sqrt(var))
		return (img + noise) * scaling_factor


class NormalNoiseModel(NoiseModel):
	"""
	Normal approximation used by Dartable database
	https://www.darktable.org/2012/12/profiling-sensor-and-photon-noise/
	"""
	def __init__(self):
		super(NormalNoiseModel, self).__init__()
		import json
		with open('darktable.json') as f:
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

	def simulate(self, img, make_str, model_str, iso, exp):
		# Darktable parameters are for normalized images
		scaling_factor = img.max()
		img = img / scaling_factor

		a, b, g = self.get_profile(make_str, model_str, iso)
		t = float(exp)

		# Rescale image
		var = (img*a[None,None,:] + b[None,None,:]) / (t*g)**2
		noise = np.random.normal(scale=np.sqrt(var))
		return (img + noise) * scaling_factor
