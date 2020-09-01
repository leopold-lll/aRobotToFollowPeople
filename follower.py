import datetime
import cv2

from utils.loadVideo import LoadVideo
from utils.storeVideo import StoreVideo
from utils.model import ResNet50, GoogleNet	#imgClassification
from utils.model import YOLOv3, MobileNetSSD	#objDetection
from utils.dataset import DatabaseOfEncodings	#preComputed-imgEncodings
from utils.knn import MyKNN

from utils.model import showDetections

class Follower:

	def __init__(self, 
			  detectorName:   str="ssd",
			  trackerName:    str="csrt",
			  classifierName: str="resNet50", 

			  sourceFPS:      int=30,
			  sourceVideo:    str="0", 
			  destVideo:	  str=None,

			  slowStartPhase: int=5000,
			  useCuda:        bool=False,
			  showSteps:	  bool=False
			  ):

		""" 
		The follower class aims to detect and track in realtime (on a webcam or on a video) the main subject there. 
		
		Parameters: 
		detectorName (str):		The name of the detector   to be used, choises: 'yolo' or 'ssd'
		trackerName (str):		The name of the tracker    to be used, choises: 'csrt', 'kcf' or not suggested: 'boosting', 'mil', 'tld', 'medianflow', 'mosse'
		classifierName (str):	The name of the classifier to be used, choises: 'resNet50' or 'googleNet'

		sourceFPS (int):	The fps ratte of the source video file (typically =30 for webcam, and =15 for shelfy)
		sourceVideo (str):	Path to the source video file, if webcam chosen, set: '0'
		destVideo (str):	Path for the output video file. If None no output will be generated.

		slowStartPhase (int):	The number of milliseconds of phase 1 lasting
		useCuda (bool):			Flag to use a GPU backend to speed-up the DNN computations
		showSteps (bool):		Flag to show additional information useful for debug, set to False for high performances.ù
								The processed frames are shown only if showSteps=True and destVideo is not None
		"""
		self.startTime = None
		self.slowStartPhase = slowStartPhase
		self.phase = 0
		self.useCuda = useCuda
		self.loop = 0
		self.detected = False
		self.showSteps = showSteps

		### Load objDetection
		self.detector = None		# YOLOv3 or MobileNetSSD
		if	 detectorName=="yolo":
			self.detector = YOLOv3(useCuda=self.useCuda) 
		elif detectorName=="ssd":
			self.detector = MobileNetSSD(useCuda=self.useCuda)
		else:
			print("[INFO] Warning: no detector chosen...")

		### Load objTracker
		#generic overview of openCV tracker: https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
		self.OBJECT_TRACKERS = {	
			"csrt": cv2.TrackerCSRT_create,
			"kcf": cv2.TrackerKCF_create,
			"boosting": cv2.TrackerBoosting_create,
			"mil": cv2.TrackerMIL_create,
			"tld": cv2.TrackerTLD_create,
			"medianflow": cv2.TrackerMedianFlow_create,
			"mosse": cv2.TrackerMOSSE_create
		}
		self.tracker = None
		if trackerName in self.OBJECT_TRACKERS:
			self.trackerName = trackerName
			self.tracker = self.OBJECT_TRACKERS[self.trackerName]()
		else:
			print("[INFO] Warning: no tracker chosen...")


		### Load imgClassification
		self.classifier = None	# ResNet50 or GoogleNet
		if	 classifierName=="resNet50":
			self.classifier = ResNet50(useCuda=self.useCuda) 
		elif classifierName=="googleNet":
			self.classifier = GoogleNet(useCuda=self.useCuda)
		else:
			print("[INFO] Warning: no classifier chosen...")

		### Load EncodeDatabase
		self.LABELS = { 0: "negative", 1: "subject" }
		self.dbEnc = DatabaseOfEncodings(classifierName, returnPath=False)
		#encs = dbEnc.getNEncodings(20)	#usage: get 20 encodings

		### Create KNN classifier
		self.knn = MyKNN()


		### Init LoadVideo
		self.sourceVideo = sourceVideo
		self.sourceFPS   = sourceFPS
		self.streamIn = LoadVideo(source=sourceVideo, simulateRealtime=sourceFPS)

		### Init StoreVideo
		if destVideo is not None:
			self.destVideo = destVideo
			self.streamOut = StoreVideo(destVideo, show=self.showSteps, fps=10)


		# check if everything is ok
		if	self.detector is None	or self.tracker is None or \
			self.classifier is None or self.dbEnc is None or \
			self.streamIn is None	or self.streamOut is None:
			print("[INFO] Warning: wrong initialization of Follower...")
			return None


	def follow(self) -> "point: tuple":
		"""
		The follow function grab a frame from the default source and then process it according to two phases:
		- slowStart phase: the algorithms (KNN) lear how to recognise the main subject 
		- follow phase: the algorithms with the accumulated knowledge, track the subject in realtime

		Returns: 
		tuple: The coordinations of the center of the bounding box of the main subject.
		"""
		#grab a frame from the input video/camera
		frame = self.streamIn.read()
		detectedPosition = None

		if frame is not None:
			#manage the slowStart phase (first k milliseconds)
			if self.phase == 0:	#phase 0=slowStart
				if self.startTime is None:
					#set the startup
					#todo: consider that the first run require almost 1.5 sec...
					self.startTime = datetime.datetime.now()

				# compute elapsed time
				elapsed = int((datetime.datetime.now() - self.startTime).total_seconds() *1000)
				if self.showSteps:
					print("elapsed", elapsed/1000, "seconds")
				#if it is the case perform change of phase
				if elapsed > self.slowStartPhase:
					self.phase = 1

				#normal walkthrough of phase 0
				detectedPosition = self.__slowStartPhase(frame)
		
			else: #phase 1=follow phase
				#normal walkthrough of phase 1
				detectedPosition = self.__followPhase(frame)
		
		if self.showSteps:
			print("fps:{:.2f}".format(self.streamIn.fps()))
		return detectedPosition


	def __slowStartPhase(self, frame: "image") -> "point: tuple":
		""" 
		The slow start phase(0)  is the first phase when the algorithm will learn the main subject in the video.
		Parameters: 
		frame (Mat): The image to be processed
  
		Returns: 
		tuple: The coordinations of the center of the bounding box of the main subject.
		"""
		detections = self.detector.detectPeopleOnly(frame)
		if self.streamOut is not None:
			newFrame = showDetections(frame, detections, 0)
			self.__writeFPS(newFrame)
			self.streamOut.addFrame(newFrame)
	
		# for logic semplicity the detection in this phase is accepted only if exactly one occours
		subjectPosition = (-1, -1)
		nDetections = len(detections)
		if nDetections == 0:
			if self.showSteps:
				print("[INFO] no detections found.")

		elif nDetections > 1:
			if self.showSteps:
				print("[INFO] found", nDetections, "and not 1.")

		else:
			#extract the feature space codification of the detected bounding box (BB)
			(x, y, w, h) = detections[0][2]
			box = frame[y:y+h, x:x+w]
			encSubj = self.classifier.feed(box)

			#fill the knn space
			self.knn.trainWithOne(encSubj, 1)	#1=subject(positive)
			self.__addOneNegativeToKNN()
			# todo: consider to add more negative or to create custom negative sample

			subjectPosition = (y+(h//2), x+(w//2))	#return the center of the bounding box

		return subjectPosition	#return the calculated position of the main subject


	def __followPhase(self, frame: "image") -> "point: tuple":
		""" 
		The follow phase(1) is the phase when the algorithm will follow based on his knowledge the main subject through the video.
		Parameters: 
		frame (Mat): The image to be processed
  
		Returns: 
		tuple: The coordinations of the center of the bounding box of the main subject.
		"""
		subjectPosition = (-1, -1)
		
		#todo: choose a better rate between detect and track
		#track once every 10 frames
		if self.loop%10 != 0:
			self.loop += 1
			if self.showSteps:
				print("tracking...")
			(success, box) = self.tracker.update(frame)
			# check to see if the tracking was a success
			if success: 
				#NB: an occlusion might return success=False even if the subject is still in the sight of the camera
				(x, y, w, h) = [int(v) for v in box]

				if self.streamOut is not None:
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
					self.__writeFPS(frame)
					self.streamOut.addFrame(frame)
				
				subjectPosition = (y+(h//2), x+(w//2))

		else:
			if self.showSteps:
				print("detecting...")
			#recognise person into the image
			detections = self.detector.detectPeopleOnly(frame)
			
			if self.streamOut:
				colors = []
			subjectFound = False
			#todo: add a method to allow only one positive subject per frame
			for (label, prob, (x, y, w, h)) in detections:
				#process each detection as a query to predict with knn the associated label.
				box = frame[y:y+h, x:x+w]
				query = self.classifier.feed(box)
				#todo: size of knn neighbours (always 5???)
				label = self.knn.predict(query, k=5)

				#choose output color according to the label
				if label==0:	# 0=negative
					if self.streamOut:
						colors.append((0, 0, 255)) #BGR format
					if self.showSteps:
						print("Query added to negative")
				else:			# 1=subject
					subjectFound = True
					#todo: verify if init is sufficient or re istantiation is needed
					self.tracker = None
					self.tracker = self.OBJECT_TRACKERS[self.trackerName]()
					self.tracker.init(frame, (x, y, w, h))
					#self.tracker.init(frame, (x, y, w, h)) #old version
					subjectPosition = (y+(h//2), x+(w//2))
					if self.streamOut:
						colors.append((0, 255, 0)) #BGR format

				self.knn.trainWithOne(query, label) #add to knn

			#add one negative (even if n detections, only one can be the subject)
			if subjectFound:
				self.loop = 1
				self.__addOneNegativeToKNN()
			
			if self.streamOut is not None:
				newFrame = showDetections(frame, detections, 0, colors)
				self.__writeFPS(newFrame)
				self.streamOut.addFrame(newFrame)

		return subjectPosition	#return the calculated position of the main subject

	def __addOneNegativeToKNN(self) -> None:
		""" Get a precomputed encoding to fill the KNN space in parallel with positive and negative samples. """
		encNeg, _ = self.dbEnc.getNEncodings(1)
		self.knn.trainWithOne(encNeg[0],  0)	#0=negative

	def __writeFPS(self, frame: "Mat") -> None:
		""" Add the fps rate (as a text) on the given frame. """
		text = "fps: {:.2f}".format(self.streamIn.fps())
		cv2.putText(frame, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (2555, 255, 255), 2)
		#return frame
