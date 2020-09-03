# import the necessary packages
from imutils.video import VideoStream, FileVideoStream, FPS
#imutils: https://github.com/jrosebr1/imutils/tree/master/imutils/video
import datetime
import time
import os

class LoadVideo:
	# The class process a video file or the webcam to return a frame when requested.

	def __init__(self, source: str="0", warmUp: float=1.0, simulateRealtime: int=0):
		""" 
		Initialization function of the LoadVideo class.
  
		Parameters: 
		source (str):		The path to the video to be load, or the number of the webcam. None is considered as webcam 0.
		warmUp (float):		The number of seconds for warm up the webcam.
		simulateRealtime(int):	If the flag is 0 the frame of the video are all used, if it is greater (fps of the input video) some frames are discarded in order to simultate a real-time processing. This flag influence only file video processing. (i.e. if 0.5 seconds pass from a request to the next one so some frames can be discarded because are already "passed" and "no more available")
		  NB: this method only speedup the processing by burn some frames, if the processing is faster than the original fps of the video nothing is done.
		"""
		if source=="-1" or None:
			source="0"

		self.source = source
		self._fps = None
		
		if source.isdigit(): #loading stream form webcam
			self.realtime = True
			self.stream = VideoStream(src=int(source)).start()
			time.sleep(warmUp)	#warm up the camera
		
		else:				#loading stream form file
			self.realtime = False
			if os.path.exists(source):
				self.stream = FileVideoStream(source).start()
			else:
				print("Warning missing source file:", source)

		#make simulateRealtime 0 even according to the type of source video
		self.simulateRealtime = 0 if self.realtime or simulateRealtime==0 else simulateRealtime
		if self.simulateRealtime!=0:
			#set some parameters for the realtime simulation
			self.millisecPerFrame = 1000/self.simulateRealtime	#how many millisec each frame took
			self.totalframesDiscarded = 0						#total number of frames discarded and not processed
			self.startTime = None								#time of start processing time

	def read(self) -> "numpy.ndarray":
		""" Return a new frame of the stream. """
		if self._fps is None:
			# initialize the FPS counter
			self._fpsStart()
		else:
			# update the FPS counter
			self._fps.update()

		#burn old frame to simultate the realtime processing on video file
		if self.simulateRealtime!=0:

			#get the new frame
			if self.startTime is None:
				self.startTime = datetime.datetime.now()

			now = datetime.datetime.now()						#get actual time
			elapsed = (now-self.startTime).total_seconds()*1000	#millisecond passed from the first frame captured

			oldDiscard = self.totalframesDiscarded							#total number of discarded frames
			self.totalframesDiscarded = int(elapsed//self.millisecPerFrame)	#total number of frames to discard (from begin)

			nFramesToDiscard = self.totalframesDiscarded - oldDiscard - 1	#how many frames left to burn
			for _ in range(nFramesToDiscard):

				#update the FPS counter (it seems like realtime)
				#self._fps.update()			#delete this row to measure the real fps...

				discard = self.stream.read()#burn already gone frames
				if discard is None:
					return None				#corner case: reach the end of the file video
				
		# get the frame that will be retrieved
		return self.stream.read()

	def _fpsStart(self) -> None:
		""" Start the FramePerSecond calculator. """
		self._fps = FPS().start()

	def elapsed(self):
		""" Compute the elapsed processing time. """
		return self._fps.elapsed()

	def fps(self, update: bool=False) -> float:
		""" Compute the fps rate of the reading processing time. 
		Parameters: 
		update (bool):	If update the count of loops or simply, retrieve the fps rate, of the frames taken from the stream.
		"""
		if self._fps is not None:
			if update:
				self._fps.update()
			self._fps.stop()
			return self._fps.fps()
		print("Warning: fps calculator not initialized.")
		return(0.0)

	def fpsOfSource(self) -> float:
		""" Return the original fps rate of the source video/webcam. """
		# To capture the fps rate of a video file the command should be: double fps = openedvideo.get(CV_CAP_PROP_FPS)
		# as stated here: https://answers.opencv.org/question/3294/get-video-fps-with-videocapture/
		#if self.realtime:
		#	print("todo: get fps of webcam")
		#else:
		#	return openedvideo.get(CV_CAP_PROP_FPS)
		print("todo: function not implemented yet.")

	def release(self) -> None:
		""" Release the all the resources used. """
		self.stream.stop()


#from storeVideo import StoreVideo
#if __name__ == "__main__":
#	lv = LoadVideo()
#	sv = StoreVideo("pippo", show=True, fps=80)

#	# Loop to capture and store the frames
#	end = False
#	while not end:
#		end = sv.addFrame(lv.read())
#	print("\n\nEND\n\n")