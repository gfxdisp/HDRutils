import subprocess as sp, logging, os

logger = logging.getLogger(__name__)

class DSLR(object):
	"""
	Control a DSLR camera connected via USB. Requires gphoto2 to communicate.
	"""
	def __init__(self, ext='.arw', test=True):
		out = sp.check_output(['gphoto2', '--auto-detect'])
		self.ext = ext
		if test:
			logger.warning(f'gphoto2 init returned:\n{out.decode()}')
			gp_command = f'gphoto2 --capture-image-and-download'
			logger.info(f'gphoto2 command: {gp_command}')

			try:
				sp.call([gp_command], shell=True)
				file = 'capt0000' + self.ext
				if not os.path.isfile(file):
					raise Exception('File does not exist. Check camera and extension provided')
				sp.call(['rm', 'capt0000' + self.ext])
			except Exception as e:
				logger.error(e)

	def set_shutter_speed(self, shutter_speed):
		if not hasattr(self, 'shutter_speed') or shutter_speed != self.shutter_speed:
			gp_command = f'gphoto2 --set-config /main/capturesettings/shutterspeed={str(shutter_speed)}'
			self.shutter_speed = shutter_speed

			logger.info(f'gphoto2 command: {gp_command}')
			sp.call([gp_command], shell=True)


	def capture_image(self, filename, shutter_speed='1', aperture='5.6', iso='100'):
		raw_file = filename + self.ext
		logger.info(f'Setting the following: exp:{shutter_speed}, iso:{iso}, aperture:{aperture}')
		gp_command = 'gphoto2'

		# Update camera parameters only as needed
		if not hasattr(self, 'shutter_speed' ) or shutter_speed != self.shutter_speed:
			gp_command += f' --set-config /main/capturesettings/shutterspeed={str(shutter_speed)}'
			self.shutter_speed = shutter_speed

		if not hasattr(self, 'iso' ) or iso != self.iso:
			gp_command += f' --set-config /main/imgsettings/iso={str(iso)}'
			self.iso = iso

		if not hasattr(self, 'aperture' ) or aperture != self.aperture:
			gp_command += f' --set-config /main/capturesettings/f-number={str(aperture)}'
			self.aperture = aperture

		gp_command += ' --capture-image-and-download --force-overwrite'

		logger.info(f'gphoto2 command: {gp_command}')
		sp.call([gp_command], shell=True)

		logger.info(f'Captured image: {raw_file}')
		sp.call(['mv', 'capt0000' + self.ext, raw_file])

	def capture_HDR_stack(self, filename, exposures,  aperture='5.6', iso='100'):
		"""
		Capture a stack of exposure modulated RAW images at a constant aperture and gain.
		:filename: Output file prefix. Images to be named 'filename_0.ext', 'filename_1.ext', ...
		:exposures: Log-spaced exposure times. Ensure that all provided values are valid settings
		:aperture: Camera f-stop (currently a constant for entire stack)
		:iso: Camera ISO to control gain (currently a constant for entire stack)
		"""
		for i, e in enumerate(exposures):
			self.capture_image(filename + '_' + str(i), shutter_speed=e, aperture=aperture, iso=iso)
