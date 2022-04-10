import subprocess, logging

logger = logging.getLogger(__name__)

class DSLR(object):
	def __init__(self):
		out = subprocess.check_output(['gphoto2', '--auto-detect'])
		logger.warning(f'gphoto2 init returned:\n{out.decode()}')

	def set_shutter_speed(self, shutter_speed):
		if not hasattr(self, 'shutter_speed') or shutter_speed != self.shutter_speed:
			gp_command = f'gphoto2 --set-config /main/capturesettings/shutterspeed={str(shutter_speed)}'
			self.shutter_speed = shutter_speed

			logger.info(f'gphoto2 command: {gp_command}')
			subprocess.call([gp_command], shell=True)


	def capture_image(self, filename, shutter_speed='1', aperture='5.6', iso='100'):
		raw_file = filename + '.arw'
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
		subprocess.call([gp_command], shell=True)

		logger.info(f'Captured image: {raw_file}')
		subprocess.call(['mv', 'capt0000.arw', raw_file])

	def capture_HDR_stack(self, filename, exposures,  aperture='5.6', iso='100'):
		for i, e in enumerate(exposures):
			self.capture_image(filename + '_' + str(i), shutter_speed=e, aperture=aperture, iso=iso)
