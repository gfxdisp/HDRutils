import subprocess as sp, logging, os
import HDRutils

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


try:
	from pyueye import ueye
except ImportError:
	logger.info('Install ueye (https://en.ids-imaging.com/software.html) and pyueye')

class IDSUeyeCamera(object):

	def __init__(self, cam_id=0, bits=12):
		self.hCam = ueye.HIDS(cam_id)
		self.sInfo = ueye.SENSORINFO()
		self.pcImageMemory = ueye.c_mem_p()
		self.memID = ueye.int()
		self.bitsPerPixel = bits
		self.colorMode = ueye.IS_CM_SENSOR_RAW12
		self.fileParams = ueye.IMAGE_FILE_PARAMS()

		# Start the driver and establish the connection to the camera
		nRet = ueye.is_InitCamera(self.hCam, None)
		if nRet != ueye.IS_SUCCESS:
			print("is_InitCamera ERROR")

		# Query additional information about the sensor type used in the camera
		ueye.is_GetSensorInfo(self.hCam, self.sInfo)
		self.width = self.sInfo.nMaxWidth
		self.height = self.sInfo.nMaxHeight

		# Set the desired color mode
		ueye.is_SetColorMode(self.hCam, self.colorMode)

		self.allocate_memory()

	# TODO: Maybe OpenCV can access the image directly 
	# https://stackoverflow.com/questions/19120198/ueye-camera-and-opencv-memory-access
	# https://docs.opencv.org/2.4.13.2/modules/core/doc/old_basic_structures.html#cv.CreateImageHeader
	def allocate_memory(self):
		# Allocate an image memory for a single image
		nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.bitsPerPixel, self.pcImageMemory, self.memID)
		if nRet != ueye.IS_SUCCESS:
			print("is_AllocImageMem ERROR")

		# Make the specified image memory the active memory
		nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.memID)
		if nRet != ueye.IS_SUCCESS:
			print("is_SetImageMem ERROR")

		# Set up the parameters required for storing images on the PC
		self.fileParams.nFileType = ueye.IS_IMG_PNG
		self.fileParams.ppcImageMem = None
		self.fileParams.pnImageID = None
		self.fileParams.nQuality = 0

		# Set Frame rate
		targetFPS = ueye.double(1) # insert here which FPS you want
		actualFPS = ueye.double(0)
		nret = ueye.is_SetFrameRate(self.hCam,targetFPS,actualFPS)

	def capture_image(self, filename, exposure=10, gain=0, black_level=255):
		nRet = ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.double(0), ueye.double(0))
		nRet += ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.double(0), ueye.double(0))
		if nRet != ueye.IS_SUCCESS:
			print("is_SetAutoParameter ERROR")

		# Set exposure time in ms, and gain in %
		targetEXP = ueye.c_double(exposure)
		nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, targetEXP, 8)
		nRet += ueye.is_SetHardwareGain(self.hCam, gain, 0, 0, 0)
		if nRet != ueye.IS_SUCCESS:
			print("is_Exposure or is_SetHardwareGain ERROR")

		# Set black level of camera
		blacklevel_offset = ueye.uint(black_level)
		blacklevel_auto = ueye.uint(ueye.IS_AUTO_BLACKLEVEL_OFF)
		nRet = ueye.is_Blacklevel(self.hCam, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET, blacklevel_offset, ueye.sizeof(blacklevel_offset))
		nRet += ueye.is_Blacklevel(self.hCam, ueye.IS_BLACKLEVEL_CMD_SET_MODE, blacklevel_auto, ueye.sizeof(blacklevel_auto))
		if nRet != ueye.IS_SUCCESS:
			print("Error setting black level")

		# Take a snapshot
		nRet = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)
		nRet = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)
		if nRet != ueye.IS_SUCCESS:
			print("Error capturing image")

		self.fileParams.pwchFileName = filename
		nRet = ueye.is_ImageFile(self.hCam, ueye.IS_IMAGE_FILE_CMD_SAVE, self.fileParams, ueye.sizeof(self.fileParams))
		if nRet != ueye.IS_SUCCESS:
			print('Error saving image', filename)
			exit(1)
		# print('Image', filename, 'captured and saved')

	def __del__(self):
		# Release the image allocated image memory
		ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.memID)

		# Disable the camera handle and exit
		ueye.is_ExitCamera(self.hCam)