# import the necessary packages
from imutils.video import VideoStream
import cv2

class StoreVideo:
	# The class accept a sequence of frames and store them as a video.

	def __init__(self, output: str="out", show: bool=False, fps: int=50, width: int=None):
		""" 
		Initialization function of the StoreVideo class.
  
		Parameters: 
		output (str):		The output file location.
		show (bool):		If to show the processd frames or not. 
		fps (int):			The fps rate of generated video.
		width (int):		The width of the output frame after rescaling. If None no rescale is performed.
		"""
		#MPEG -> avi
		#XVID -> avi
		
		#FMP4 -> mp4
		#MJPG -> mp4
		self.output = ".".join([output, "avi"])
		self.codec = "MPEG"
		self.fourcc = cv2.VideoWriter_fourcc(*self.codec)	#extension standard
		
		# fourcc stands for "Four Character Code" it is a representation of the video file extensions. 
		# more info at: http://www.fourcc.org/fourcc.php
		# discussion at: https://answers.opencv.org/question/68262/how-to-make-a-good-long-video-capture-with-videowriter/
		self.show = show
		self.width = width
		self.fps = fps

		self.writer = None
		self.h = None
		self.w = None

	def addFrame(self, frame: "numpy.ndarray", fps: int=None, isColor: bool=True) -> bool:
		""" 
		Add the given frame to the video.
  
		Parameters: 
		frame (numpy.ndarray):	The frame of the camera that need to be add to the video.
		fps (int):				The fps rate of generated video (it will be consideronly at the first call).
		isColor (bool):			If the passed frame is colored, False means grayscale.

		Returns:
		bool: True if the user ask to stop ('q' button when frames are shown), False otherwise.
		"""
		# grab the frame from the video stream and resize it to have a maximum width of 300 pixels
		if self.width is not None:
			frame = imutils.resize(frame, width=self.width)

		# check if the writer is None
		if self.writer is None:
			if fps is not None:
				self.fps = fps
			
			# store the image dimensions, initialize the video writer, and construct the zeros array
			(self.h, self.w) = frame.shape[:2]
			self.writer = cv2.VideoWriter(self.output, self.fourcc, self.fps, (self.w, self.h), isColor)

		# write the output frame to file
		self.writer.write(frame)

		# show the frames
		if self.show:
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				return True
		return False

	def release(self) -> None:
		""" Release the file writer. """
		self.writer.release()


#if __name__ == "__main__":
#	sv = StoreVideo("pippo")
	
#	# Open camera and warm it up
#	vs = VideoStream(src=0).start()
#	time.sleep(2.0)

#	# Loop to capture and store the frames
#	end = False
#	while not end:
#		end = sv.addFrame(vs.read())
#	print("\n\nEND\n\n")