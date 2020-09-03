from scipy import spatial
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
	def setHyperparam(self, 
			slowStartPhase: int=5000,
			trackOverDetect:int=10,
			k:				int=5,
			driftRatio:		float=0.02,
			driftTollerance:int=30
			):
		""" 
		This function aim to set all the hyperparams that control the logic of the algorithm.
		
		Parameters: 
		slowStartPhase  (int): The lenght (in milliseconds) of the first phase.
		trackOverDetect (int): How many frames contain the loop of 1 detection and the rest track.
		k 				(int): The K value of knn. K at the moment is a constant value it might be change into a probability or into a list of values with a majority score.
		driftRatio 	  (float): The multiply factor (pixel/milliseconds) of the drifting during the tracking.
		driftTollerance (int): The initial ammount of the drift value. It's important in case of very small track periods.
		"""

		self.trackOverDetect = trackOverDetect
		self.slowStartPhase = slowStartPhase
		self.K = k

		#drift control parameters
		self.driftRatio = driftRatio			#measure of pixel/millisecond of drifting effect
		self.driftTollerance = driftTollerance	#tollerace on small driftPeriods
		self.lastW = 100	#scaling factor according to BB width (one dimension, basecase=100) to simulate depth 
		# I assume that every millisec the measure drift by x pixel from real position
		# todo: this measure may be influenced even from the BB avg size that mainly map the subject distance.

	def __init__(self, 
			  detectorName:   str="ssd",
			  trackerName:    str="csrt",
			  classifierName: str="resNet50", 

			  sourceFPS:      int=30,
			  sourceVideo:    str="0", 
			  destVideo:	  str=None,

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
		self.setHyperparam()
		self.startTime = None
		self.phase = 0
		self.useCuda = useCuda
		self.loop = 0
		self.detected = False
		self.showSteps = showSteps

		#positions of the robot in the image
		self.idlePosition = (-1, -1)
		self.lastKnownPosition = None	#A tuple containing the last measured position of the robot

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
					self.startDrifting = self.startTime

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

		# update the last known position if it is not the idle one
		if detectedPosition != self.idlePosition:
			self.lastKnownPosition = detectedPosition
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
			showDetections(frame, detections, 0)
			self.__writeFPS(frame)
			self.streamOut.addFrame(frame)
	
		# for logic semplicity the detection in this phase is accepted only if exactly one occours
		subjectPosition = self.idlePosition
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

			subjectPosition = centerBB(x, y, w, h)	#return the center of the bounding box

		return subjectPosition	#return the calculated position of the main subject


	def __followPhase(self, frame: "image") -> "point: tuple":
		""" 
		The follow phase(1) is the phase when the algorithm will follow based on his knowledge the main subject through the video.
		Parameters: 
		frame (Mat): The image to be processed
  
		Returns: 
		tuple: The coordinations of the center of the bounding box of the main subject.
		"""
		subjectPosition = self.idlePosition
		
		#TRACK once every x frames
		if self.loop%self.trackOverDetect != 0:
			self.loop += 1
			if self.showSteps:
				print("tracking...")
			(success, box) = self.tracker.update(frame)
			# check to see if the tracking was a success
			if success: 
				#NB: an occlusion might return success=False even if the subject is still in the sight of the camera
				(x, y, w, h) = [int(v) for v in box]
				self.lastW = w

				if self.streamOut is not None:
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
				
				subjectPosition = centerBB(x, y, w, h)

		#DETECT
		#todo: manage the case when the leader is classified as negative
		else:
			if self.showSteps:
				print("detecting...")
			#recognise person into the image
			detections = self.detector.detectPeopleOnly(frame)
			
			if self.streamOut:
				colors = []
			leadersFound = []

			#clac the drift radius tollerance according to the ratio and the period gone
			driftPeriod = int((datetime.datetime.now() - self.startDrifting).total_seconds() *1000)
			driftRadius = int(driftPeriod*self.driftRatio*(self.lastW/100) + self.driftTollerance) 
			print("driftRadius:", driftPeriod*self.driftRatio, "*", (self.lastW/100), "+", self.driftTollerance, "=", driftRadius)
			for (i, (_label, _prob, (x, y, w, h))) in enumerate(detections):
				#process each detection as a feature vector (used here as a point for knn)
				box = frame[y:y+h, x:x+w]
				featuresPoint = self.classifier.feed(box)

				#filter out too far detection according to the drift radius
				dist = spatial.distance.euclidean(self.lastKnownPosition, centerBB(x, y, w, h))
				if driftRadius < dist:
					if self.showSteps:
						print("\n\ndetection out of bound of drift radius. Set as negative")
						print("driftRadius:", driftRadius)
						print("spatial.distance.euclidean(self.lastKnownPosition, centerBB(x, y, w, h)):", dist)
						print("self.lastKnownPosition:", self.lastKnownPosition)
						print("centerBB(x, y, w, h):", centerBB(x, y, w, h))
						print("\n")
					label = 0
				else:
					if self.showSteps:
						print("Choice with KNN")
					label = self.knn.predict(featuresPoint, k=self.K)

				#choose output color according to the label
				if self.streamOut:
						colors.append((0, 0, 255)) #BGR format
				if label==0:	# 0=negative
					if self.showSteps:
						print("featuresPoint added to negative")
					self.knn.trainWithOne(featuresPoint, 0) #add to knn
				else:			# 1=leader
					leadersFound.append(i)

			if len(leadersFound) == 0:
				leader = None
				if self.showSteps:
					print("No leader detected...")
			#todo: move the multiple leaders check to a function
			elif len(leadersFound) > 1:
				if self.showSteps:
					print("More than one leader detected...")

				#choose the closer leader to the last known position.
				#that one will be chosen as the leader
				bestI = -1
				bestDist = None
				for i in leadersFound:
					_, _, (x, y, w, h) = detections[i]
					dist = spatial.distance.euclidean(self.lastKnownPosition, centerBB(x, y, w, h)) 
					bestDist = dist if bestDist is None else min(bestDist, dist)
					if dist==bestDist:
						bestI = i
				leader = bestI
			else:
				leader = leadersFound[0]

			if leader is not None:
				_, _, (x, y, w, h) = detections[leader]

				# reinitialize the tracker. A simple initialization is not sufficient
				self.tracker = None
				self.tracker = self.OBJECT_TRACKERS[self.trackerName]()
				self.tracker.init(frame, (x, y, w, h))

				# compute subject position
				subjectPosition = centerBB(x, y, w, h)


				#tmp: compute confidence distance from last tracked position to first detect position
				dist = spatial.distance.euclidean(self.lastKnownPosition, subjectPosition)

				with open("imagesOut/confidenceDist.txt", 'a') as f:
					f.write("{:.2f}".format(dist))
					f.write("\t")
					f.write(str(driftPeriod))	#elapsed millisec
					f.write("\n")
				self.startDrifting = datetime.datetime.now() #tmp (next round drift period)


				if self.streamOut:
					colors[i] = (0, 255, 0) #BGR format



			#todo: add even if a negative already occour?
			#add one negative (even if n detections, only one can be the subject)
			if leader is not None: #aka len>0
				self.loop = 1
				# I add a precomputed negative sample only if there is only the leader in the frame
				# to balance positive and negative.
				if len(detections)<=1:
					self.__addOneNegativeToKNN()
			
			if self.streamOut is not None:
				showDetections(frame, detections, 0, colors)
				cv2.circle(frame, self.lastKnownPosition, driftRadius, color=(0, 255, 255), thickness=1)
				cv2.circle(frame, self.lastKnownPosition, 5, color=(0, 0, 0), thickness=2)

		if self.streamOut is not None:
			self.__writeFPS(frame)
			self.streamOut.addFrame(frame)

		return subjectPosition	#return the calculated position of the main subject

	def __addOneNegativeToKNN(self) -> None:
		""" Get a precomputed encoding to fill the KNN space in parallel with positive and negative samples. """
		encNeg, _ = self.dbEnc.getNEncodings(1)
		self.knn.trainWithOne(encNeg[0],  0)	#0=negative

	def __writeFPS(self, frame: "Mat") -> None:
		""" Add the fps rate (as a text) on the given frame. """
		text = "fps: {:.2f}".format(self.streamIn.fps())
		cv2.putText(frame, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (2555, 255, 255), 2)

def centerBB(x, y, w, h):
	return( (x+(w//2), y+(h//2)) )

#def swapXY(p):
#	return( (p[1], p[0]) if p is not None else None )
	