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
		driftRatio 	  (float): The multiply factor (pixel/milliseconds) of the drifting during the tracking. The ratio is considered for a 100pixel wide bounding box, and a rescaling to the power of two is done to simulate closer and further person.
		driftTollerance (int): The initial ammount of the drift value. It's important in case of very small track periods.
		"""

		self.slowStartPhase = slowStartPhase
		self.trackOverDetect = trackOverDetect
		self.K = k

		#drift control parameters
		self.driftRatio = driftRatio			#measure of pixel/millisecond of drifting effect
		self.driftTollerance = driftTollerance	#tollerace on small driftPeriods
		self.lastW = 100	#scaling factor according to BB width (one dimension, basecase=100) to simulate depth 
		# I assume that every millisec the measure drift by x pixel from real position


	def setLogParam(self, 
			showLog:	bool=False,
			destImgs:	str=None,
			destVideo:	str=None,
			destFps:	int=5
			):
		""" 
		This function aim to set all the hyperparams that control the logic of the algorithm.
		
		Parameters: 
		showLog (bool):	Flag to show additional information useful for debug, set to False for high performances.
							The processed frames are shown only if showLog=True and destVideo is not None
		destImgs	  (str):	Path to the output folder for images. If None no images output will be generated.
		destVideo (str):	Path to the output video file. If None no video output will be generated.
		destFps	  (int):	The fps rate of the output file video.
		"""

		self.showLog = showLog	

		self.destImgs = destImgs
		if self.destImgs is not None:
			self.random = 10000

		### Init StoreVideo
		self.destFps   = destFps		
		self.destVideo = destVideo
		if destVideo is not None:
			self.streamOut = StoreVideo(destVideo, show=self.showLog, fps=self.destFps)
		else: 
			self.streamOut = None

	def __init__(self, 
			  detectorName:   str="ssd",
			  trackerName:    str="csrt",
			  classifierName: str="resNet50", 

			  sourceFPS:      int=30,
			  sourceVideo:    str="0", 
			  useCuda:        bool=False
			  ):

		""" 
		The follower class aims to detect and track in realtime (on a webcam or on a video) the main subject there. 
		
		Parameters: 
		detectorName (str):		The name of the detector   to be used, choises: 'yolo' or 'ssd'
		trackerName (str):		The name of the tracker    to be used, choises: 'csrt', 'kcf' or not suggested: 'boosting', 'mil', 'tld', 'medianflow', 'mosse'
		classifierName (str):	The name of the classifier to be used, choises: 'resNet50' or 'googleNet'

		sourceFPS (int):	The fps ratte of the source video file (typically =30 for webcam, and =15 for shelfy)
		sourceVideo (str):	Path to the source video file, if webcam chosen, set: '0'
		useCuda (bool):			Flag to use a GPU backend to speed-up the DNN computations
		"""

		#set other main parameters
		self.useCuda = useCuda	#set the DNN to use a GPU backend
		self.setHyperparam()	#hyperparam to cvontrol the algorithm logic
		self.setLogParam()		#deteails to save a log for the user and a later analysis

		#main flow parameters
		self.phase = 0			#which phase is running (0=slow start, 1=follow)
		self.loop = 0			#counter for switching from detection to tracker every x frames
		self.startTime = None	#startTime to measure the lenght of the first phase

		#positions of the robot in the image
		self.idlePosition = (-1, -1)	#position returned if the leader will not be located
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
		self.dbEnc = DatabaseOfEncodings(classifierName, returnPath=True)

		### Create KNN classifier
		self.knn = MyKNN()

		### Init LoadVideo
		self.sourceVideo = sourceVideo
		self.sourceFPS   = sourceFPS
		self.streamIn = LoadVideo(source=sourceVideo, simulateRealtime=sourceFPS)

		# check if everything is ok
		if	self.detector is None	or self.tracker is None or \
			self.classifier is None or self.dbEnc is None or \
			self.knn is None		or self.streamIn is None or \
			(self.destVideo is not None and self.streamOut is None): #check streamOut only if required from user
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
					self.startTime = datetime.datetime.now()
					self.startDrifting = self.startTime

				# compute elapsed time
				elapsed = int((datetime.datetime.now() - self.startTime).total_seconds() *1000)
				if self.showLog:
					print("elapsed", elapsed/1000, "seconds")
				#if it is the case perform change of phase
				if elapsed > self.slowStartPhase:
					self.phase = 1

				#normal walkthrough of phase 0
				detectedPosition = self.__slowStartPhase(frame)
		
			else: #phase 1=follow phase
				#normal walkthrough of phase 1
				detectedPosition = self.__followPhase(frame)
		
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
	
		# for logic semplicity the detection in this phase is accepted only if exactly one occours
		subjectPosition = self.idlePosition
		nDetections = len(detections)
		if nDetections == 0:
			if self.showLog:
				print("[INFO] no detections found.")

		elif nDetections > 1:
			if self.showLog:
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

			if self.destImgs is not None:
				self.random += 1
				cv2.imwrite("".join([self.destImgs, str(self.random), ".jpg"]), box)

		
		#draw on the frame output for the user (with both the suggestions below)
		#NB: it is fundamental to draw on the frame AFTER that the classifier has elaborated the frame itself, or in case of elaboration before is neccessary to draw on a COPY of the frame.
		#If this do not happen the classifier works on a drawn frame, and it will learn the "user friendly draw" and not the frame
		if self.streamOut is not None:
			newFrame = frame.copy()
			showDetections(newFrame, detections, 0)
			self.__writeFPS(newFrame)
			self.streamOut.addFrame(newFrame)

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
		
		#TRACK 
		if self.loop%self.trackOverDetect != 0:
			self.loop += 1
			if self.showLog:
				print("\nTracking...")
			(success, box) = self.tracker.update(frame)
			# check to see if the tracking was a success
			if success: 
				#NB: an occlusion might return success=False even if the subject is still in the sight of the camera
				(x, y, w, h) = [int(v) for v in box]
				subjectPosition = centerBB(x, y, w, h)
				self.lastW = w	#neccessary for drift measure (simulate depth of BB according to the width of the BB itself)

				if self.streamOut is not None:
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

		#DETECT once every x frames
		#todo: manage the case when the leader is classified as negative. how??? -> it's the weak point of this algorithm....
		else:
			if self.showLog:
				print("\nDetecting...")

			#recognise people into the image
			detections = self.detector.detectPeopleOnly(frame)
			
			#initialaze support lists
			leadersFound = []
			features = []
			labels = []
			if self.streamOut:
				colors = []

			#calc the drift radius tollerance according to the ratio, the period gone and the BB size
			driftPeriod = int((datetime.datetime.now() - self.startDrifting).total_seconds() *1000)
			driftRadius = int(driftPeriod*self.driftRatio*((self.lastW/100)**2) + self.driftTollerance) 

			#process each detection
			for (i, (__classLabel, __prob, (x, y, w, h))) in enumerate(detections):
				#convert a detection (aka BB) to feature vector
				box = frame[y:y+h, x:x+w] 
				featureVector = self.classifier.feed(box)

				#filter out detection too far away according to the drift radius
				drift = spatial.distance.euclidean(self.lastKnownPosition, centerBB(x, y, w, h))
				if driftRadius < drift:
					#person too far away
					label = 0
					#if self.showLog:
					#	print("\n\nDetection out of bound of drift radius. Set as negative")
					#	print("driftRadius:", driftRadius)
					#	print("drift: spatial.distance.euclidean(self.lastKnownPosition, centerBB(x, y, w, h)):", drift)
					#	print("self.lastKnownPosition:", self.lastKnownPosition)
					#	print("centerBB(x, y, w, h):", centerBB(x, y, w, h))
					#	print("\n")

				else:
					if self.showLog:
						print("Choice with KNN")
					#add the featureVector to KNN as a point
					label = self.knn.predict(featureVector, k=self.K)

				if label==1:	# 1=leader
					leadersFound.append(i)

				#update support lists
				features.append(featureVector)
				labels.append(label)

			# Manage the case of multiple leaders detected (only one can exist!)
			leader = None
			if len(leadersFound) == 0:
				if self.showLog:
					print("No leader detected...")
			elif len(leadersFound) == 1:
				leader = leadersFound[0]
			else:
				if self.showLog:
					print("More than one leader detected...")

				#choose the closer leader, to the last known position, as the official one
				bestI = -1
				bestDist = None
				for i in leadersFound:
					_, _, (x, y, w, h) = detections[i]
					dist = spatial.distance.euclidean(self.lastKnownPosition, centerBB(x, y, w, h)) 
					bestDist = dist if bestDist is None else min(bestDist, dist)
					if dist==bestDist:
						bestI = i
				#remember the i relative to the lowest distance
				leader = bestI

			#for each person previously found (now identified by featureVector and label)
			for (fv, l) in zip(features, labels):

				#choose output color according to the label
				self.knn.trainWithOne(fv, l) #add to knn
				if self.destImgs is not None:
					self.random += 1
					cv2.imwrite("".join([self.destImgs, "neg-" if l==0 else "", str(self.random), ".jpg"]), box)
				if self.streamOut is not None:
					colors.append( (0, 0, 255) if l==0 else (0, 255, 0) ) #BGR format

				if l==0: # 0=negative 
					if self.showLog:
						print("FeatureVector added to negative")

				#thanks to leader selection exist exactly one leader (loop enter here only once)
				else:	 # 1=positive
					_, _, (x, y, w, h) = detections[leader]
				
					#leader found: next frame will be used for tracking
					self.loop = 1
					self.startDrifting = datetime.datetime.now()
					# reinitialize the tracker. A simple initialization is not sufficient it need to be re-istantiated
					self.tracker = None
					self.tracker = self.OBJECT_TRACKERS[self.trackerName]()
					self.tracker.init(frame, (x, y, w, h))

					# compute subject position
					subjectPosition = centerBB(x, y, w, h)

					# add one precomputed negative sample to KNN, because only the leader exist. 
					# the goal is to balance positive and negative.
					if len(labels)==1:
						self.__addOneNegativeToKNN()

					#tmp: compute confidence distance from last tracked position to first detect position
					#usefull to decide the right value of the "driftRatio" hyperparameter
					#if self.showLog:
					#	dist = spatial.distance.euclidean(self.lastKnownPosition, subjectPosition)

					#	with open("imagesOut/confidenceDist.txt", 'a') as f:
					#		f.write("{:.2f}".format(dist))
					#		f.write("\t")
					#		f.write(str(driftPeriod))	#elapsed millisec
					#		f.write("\t")
					#		f.write(str(int(w)))		#width of BB
					#		f.write("\n")

			
			if self.streamOut is not None:
				#NB: the draw on the frame MUST be done after the classifier has analised it, if not tests will be compromised
				showDetections(frame, detections, 0, colors)
				cv2.circle(frame, self.lastKnownPosition, driftRadius, color=(0, 255, 255), thickness=1)
				cv2.circle(frame, self.lastKnownPosition, 5, color=(0, 0, 0), thickness=2)
	
		#writing for both tracking and detetion
		if self.streamOut is not None:
			self.__writeFPS(frame)
			self.streamOut.addFrame(frame)

		return subjectPosition	#return the calculated position of the leader

	def __addOneNegativeToKNN(self) -> None:
		""" Get a precomputed encoding to fill the KNN space in parallel with positive and negative samples. """
		encNeg, _paths = self.dbEnc.getNEncodings(1)
		self.knn.trainWithOne(encNeg[0],  0)	#0=negative

		if self.destImgs is not None:
			img = cv2.imread(_paths[0][1:])
			cv2.imwrite("".join([self.destImgs, _paths[0].split("/")[-1]]), img)

	def __writeFPS(self, frame: "Mat") -> None:
		""" Add the fps rate (as a text) on the given frame. """
		text = "fps: {:.2f}".format(self.streamIn.fps())
		cv2.putText(frame, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (2555, 255, 255), 2)

def centerBB(x, y, w, h):
	""" Compute the center of the bounding box given. """
	return( (x+(w//2), y+(h//2)) )

#def swapXY(p):
#	""" Swap the first and second value of a tuple. """
#	return( (p[1], p[0]) if p is not None else None )
	