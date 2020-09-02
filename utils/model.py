# import the necessary packages
from abc import ABC # ABC = Abstract Method Class
import numpy as np
import cv2
import random #tmp

################################################################################################################
############################     Model Super Class        ######################################################
################################################################################################################
class Model(ABC):
	### Variables
	modelPath:  str=None		#path to the model
	modelPath2: str=None		#path to the second part of the model
	net: "cv2.dnn.net"=None		#the opencv network model
	# objDetection
	namesPath:  str=None		#path to the file where the names of detection are saved
	names:  list=None			#the list of names of detection
	# imgClassification
	layer: str=None				#the name of the layer where to end the DNN

	### Set Functions
	def __init__():
		pass

	def setNetwork(self, net: "cv2.dnn.net") -> None:
		""" Set the network given. """
		self.net = net

	def setLayer(self, layer: str=None) -> None:
		""" Set the output layer name of the network (if None the last one is chosen). """
		self.layer = layer
		
	### Main Flow Functions
	def blob(self, image) -> "blob":
		""" Create the blob form the image given. """
		pass

	def setInput(self, blob) -> None:
		""" Set the input blob for the DNN. """
		self.net.setInput(blob)

	def forward(self) -> list:
		""" Process the given input through the DNN and return the output of the specified layer of the DNN. """
		return self.net.forward(self.layer) 

	def processDnnOutput(self, output, h: int=None, w: int=None, confidence: float=None) -> "list or list of list":
		""" 
			Process the output of the DNN to standardize it.
			- for image Classification: a list of floats (the feature vector) -> AKA encoding
			- for object Detection: a list detection, each one is a list as: [label, confidence, [x, y, width, height] ]
			input variables:
			- output of DNN
			- h: height of original image
			- w: width  of original image
			- confidence to filter weak prediction
		"""
		pass
		#conversion with np.array:
		#to   NP: npList = np.array(myList)
		#from NP: myList = npList.tolist()
	
	def feed(self, image, confidence: float=0.5) -> "list or list of list":
		""" Do in one function: blob, setInput, forward pass and processDnnOutput. """
		blob = self.blob(image)
		self.setInput(blob)
		out = self.forward()
		(h, w) = image.shape[:2]
		return self.processDnnOutput(out, h, w, confidence)

	def detectPeopleOnly(self, image, confidence: float=0.5) -> "list of list":
		""" Do the same of feed but filter the result to retrieve people only. (work only for detection algorithms) """
		detections = self.feed(image, confidence)
		peopleBB = []
		for d in detections:
			if d[0]=="person":
				peopleBB.append(d)
		return peopleBB
		

	### Utils Functions
	def getLayerNames(self, show: bool=False) -> "list or list of list":
		""" Return and eventually show the list of layers of the model. """
		layerNames = self.net.getLayerNames()
		if show:
			for (i, l) in enumerate(layerNames):
				print(i, "->", l)
			print("\n")
		return(layerNames)
	
	def useCUDA(self):
		# set CUDA as the preferable backend and target
		if cv2.cuda.getCudaEnabledDeviceCount() > 0:
			print("[INFO] setting preferable backend and target to GPU and CUDA...")
			self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		else: 
			print("[WARNING] tried to setup a GPU backend but no GPU found, CPU will be used.")

################################################################################################################
############################     Object Detection         ######################################################
################################################################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>     YOLO v3                  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class YOLOv3(Model):
	def __init__(self, 
			  modelPath:  str="./models/yoloV3-coco/yolov3.cfg", 
			  modelPath2: str="./models/yoloV3-coco/yolov3.weights", 
			  namesPath:  str="./models/yoloV3-coco/coco.names",
			  layer:	  list=['yolo_82', 'yolo_94', 'yolo_106'],
			  useCuda: bool=False):

		self.modelPath  = modelPath
		self.modelPath2 = modelPath2
		self.net = cv2.dnn.readNetFromDarknet(modelPath, modelPath2)
		
		self.namesPath = namesPath
		self.names = open(namesPath).read().strip().split("\n")
		self.layer = layer

		if useCuda:
			super(YOLOv3, self).useCUDA()
	
	def blob(self, image) -> "blob":
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		return blob

	def processDnnOutput(self, output, h: int, w: int, confThresh: float=0.5, nmsThresh: float=0.3) -> "list of list":
		classes = []
		confidences = []
		boxes = []
		
		# loop over each of the layer outputs (yolo use 3 output layers)
		for oneLayerOutput in output:
			# loop over each of the detections
			for detection in oneLayerOutput:

				# extract the class ID and confidence (i.e., probability) of the current object detection
				scores = detection[5:]			# a detection has a probability(confidence) for each class (80 for coco)
				classID = np.argmax(scores)		# extract the class with highest confidence
				confidence = scores[classID]	# get the highest confidence AKA: np.max(scores)

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > confThresh:
					#todo: accept only person
					# scale the bounding box coordinates back relative to the size of the image, 
					# keeping in mind that YOLO actually returns the center (x, y)-coordinates of the bounding box 
					# followed by the boxes' width and height
					box = detection[0:4] * np.array([w, h, w, h])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					
					# update our list of bounding box coordinates, confidences and classes
					classes.append(self.names[classID])
					confidences.append(float(confidence))
					boxes.append([x, y, int(width), int(height)])	#aka: x, y, w, h
		
		#apply non-maxima suppression to suppress weak, overlapping bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThresh, nmsThresh)
		print("idxs:", idxs)
		detections = []

		if len(idxs)>0:
			idxs = idxs.flatten()
			#remove overlapping predictions selected by NMS
			detections = [ [classes[i], confidences[i], boxes[i]] for i in idxs ]

		return detections
	

#>>>>>>>>>>>>>>>>>>>>>>>>>>>     MobileNet SSD            <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class MobileNetSSD(Model):
	def __init__(self, 
			  modelPath:  str="./models/mobileNet_SSD/MobileNetSSD_deploy.prototxt", 
			  modelPath2: str="./models/mobileNet_SSD/MobileNetSSD_deploy.caffemodel", 
			  namesPath:  str="./models/mobileNet_SSD/SSDnames.txt",
			  layer:	  str=None,
			  useCuda:    bool=False):

		self.modelPath  = modelPath
		self.modelPath2 = modelPath2
		self.net = cv2.dnn.readNetFromCaffe(self.modelPath, self.modelPath2)
		
		self.namesPath = namesPath
		self.names = open(namesPath).read().strip().split("\n")
		self.layer = layer

		if useCuda:
			super(MobileNetSSD, self).useCUDA()

	def blob(self, image) -> "blob":
		blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
		return blob

	def processDnnOutput(self, output, h: int, w: int, confThresh: float=0.5) -> "list of list":
		detections = []

		# loop over the detections
		for i in np.arange(0, output.shape[2]):
			# extract the confidence (i.e., probability) associated with the prediction
			confidence = output[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
			if confidence > confThresh:
				# extract the index of the class label from the `detections`, 
				# then compute the (x, y)-coordinates of the bounding box for the object
				idx = int(output[0, 0, i, 1])
				#todo: accept only person
				
				box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				box = [	max(0, startX), 
						max(0, startY), 
						max(0, endX-startX), 
						max(0, endY-startY)]	#aka: x, y, w, h

				detections.append([ self.names[idx], confidence, box ])

		return detections


################################################################################################################
############################     Image Classification     ######################################################
################################################################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>     ResNet50 Model           <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ResNet50(Model):
	def __init__(self, 
			  modelPath: str="./models/resnet50-caffe2/resnet50-caffe2.onnx", 
			  layer: str="OC2_DUMMY_0",
			  useCuda: bool=False):

		self.modelPath = modelPath
		self.layer = layer
		self.net = cv2.dnn.readNetFromONNX(self.modelPath)
		
		if useCuda:
			super(ResNet50, self).useCUDA()
			
	def blob(self, image) -> "blob":
		#image = cv2.resize(image, (64, 128), interpolation = cv2.INTER_AREA) #not neccessary
		return cv2.dnn.blobFromImage(image=image, scalefactor=1, size=(224, 224), mean=(0.485, 0.456, 0.406))

	def processDnnOutput(self, output, _1=None, _2=None, _3=None) -> "list":
		return output[0].tolist()


#>>>>>>>>>>>>>>>>>>>>>>>>>>>     GoogleNet Model          <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class GoogleNet(Model):
	def __init__(self, 
			  modelPath:  str="./models/googleNet/bvlc_googlenet.prototxt",
			  modelPath2: str="./models/googleNet/bvlc_googlenet.caffemodel",
			  layer: str="pool5/7x7_s1",
			  useCuda: bool=False):

		self.modelPath  = modelPath
		self.modelPath2 = modelPath2
		self.layer = layer
		self.net = cv2.dnn.readNetFromCaffe(self.modelPath, self.modelPath2)

		if useCuda:
			super(GoogleNet, self).useCUDA()

	def blob(self, image) -> "blob":
		return cv2.dnn.blobFromImage(image=image, scalefactor=1, size=(224, 224), mean=(0.485, 0.456, 0.406))

	def processDnnOutput(self, output, _1=None, _2=None, _3=None) ->"list":
		vect = [ [ el[0][0] for el in enc ] for enc in output ]
		return vect[0]

################################################################################################################

def showDetections(image: "Mat", detections: "list of list", waitTime: int=1, colors: "cv::viz::Color"=[(255, 0, 0)]) -> "image":
	""" 
		This function draw on the input image rectangles of different colors according to the detections.
		Parameters: 
		image (Mat): The image to be elaborated
		detections (list of list): The coordination of the detected bounding boxes
		waitTime (int): the millisecond that each shown frame will last on the video. If 0, frames are not shown.
		colors (list of tuples): The colors assoziated with each detection.
		fps: ... todo: to add
  
		Returns: 
		image (Mat): The elaborated frame with all the detections, and text on it.
	"""
	# ensure at least one detection exists
	newImg = image.copy()
	if len(detections) > 0:
		#if len(detections) > 1:
		#	print("\nMore than one detection:")
		#	print(detections)
		#	print(colors)

		color = colors[0]

		for i in range(len(detections)):
			# extract the bounding box coordinates
			label = detections[i][0]
			prob  = detections[i][1]
			(x, y, w, h) = detections[i][2]

			# choose one color for each detection (otherwise one for all)
			if len(colors)==len(detections):
				color = colors[i]

			# draw a bounding box rectangle and label on the image
			cv2.rectangle(newImg, (x, y), (x+w, y+h), color, 2)

			text = "{}: {:.2f}%".format(label, 100*prob)
			cv2.putText(newImg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	if waitTime!=0:
		cv2.imshow("image", newImg)
		cv2.waitKey(waitTime)
	return newImg