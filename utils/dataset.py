# import the necessary packages
from abc import ABC # ABC = Abstract Method Class
from imutils import paths
import random
import os

################################################################################################################
############################     Dataset Abstract Class     ####################################################
################################################################################################################
#For abstract class look at: https://www.geeksforgeeks.org/abstract-classes-in-python/
class Dataset(ABC):

	def getLabelsAndPaths(self) -> "list of couple":
		""" Create a "list of couple" of lenght (nImgs) with key (1) the label of the person and value (2) the paths of its images. """
		pass

	def queryImgPath(self, queryNum: int=None) -> (int, str):
		""" 
			Return an image (label and path) to use as query. 
			queryNum: optionally chosen from the user: the image number to be used as query
		"""
		pass

############################     Negative People dataset functions     #########################################
class NegativePeopleDataset(Dataset):
	def __init__(self, nImgs: int=50, nTestDir: int=9) -> None:
		self.N_IMAGES = 518			#the number of images in the dataset (do not change)
		self.datasetPath = "./imagesIn/negativePeople_dataset/repeatedPeople"
		self.subFolderName = "set"
		self.setImgsGeneric = '/'.join([self.datasetPath, self.subFolderName])
		self.nTestDir = nTestDir	#how many folder of the dataset use to train (the next one is for test)
		self.nImgs = nImgs			#how many images to process
		if nImgs/nTestDir > self.N_IMAGES:
			self.nImgs = self.N_IMAGES*nTestDir
			print("Warning: not enoght images in the dataset (set to", self.nImgs, ")")

	### Private functions
	def __labelFromPath(self, pth):
		return( int(pth.split("/")[-1].split(".")[0]) )

	def __getLabelsAndPaths_fromFolder(self, folderPath, nImgs) -> "list of couple":
		labelsAndPaths = []
		imagesPaths = list(paths.list_images(folderPath))[:nImgs]
		for pth in imagesPaths:
			pth = pth.replace("\\", "/")
			label = self.__labelFromPath(pth)
			labelsAndPaths.append((label, pth))
		
		return(labelsAndPaths)

	### Public functions
	def getLabelsAndPaths(self) -> "list of couple":
		labelsAndPaths = []
		#take the same number of images from each of the 9 folders of the dataset (the 10th is used as testing folder)
		nImgsPartial = self.nImgs//self.nTestDir 

		for i in range(self.nTestDir):
			setImgs = ''.join([self.setImgsGeneric, str(i)])
			tmp = self.__getLabelsAndPaths_fromFolder(setImgs, nImgsPartial)
			[labelsAndPaths.append(el) for el in tmp]

		print("Added", nImgsPartial, "images, for each person/label:", [el[0] for el in labelsAndPaths[:nImgsPartial]], "\n")
		return(labelsAndPaths)

	def queryImgPath(self, queryNum: int=None) -> (int, str):
		setImgs = ''.join([self.setImgsGeneric, str(self.nTestDir), "/"])
		queryPath = ""
		if queryNum is None:
			#get a random image
			num = random.randint(0, self.nImgs//self.nTestDir)
			queryPath = list(paths.list_images(setImgs))[num]
			queryNum = self.__labelFromPath(queryPath)
		else:
			#get the image chosen from the user
			queryPath = "".join([setImgs, f'{queryNum:04}', ".jpg"])

		if not os.path.exists(queryPath):
			queryPath = None
		print("generated query:", (queryNum, queryPath))
		return((queryNum, queryPath))

############################     Street People dataset functions       #########################################
class StreetPeopleDataset(Dataset):
	def __init__(self, nImgs: int=20) -> None:
		self.datasetPath = "./imagesIn/streetPeople_dataset"
		self.nImgs = nImgs
		if nImgs>50:
			print("Warning: not enoght images in the dataset (set to 50)")
			self.nImgs = 50

	def getLabelsAndPaths(self) -> "list of couple":
		# get the path
		# str(os.path.sep) should be equal to '/', but for windows is not
		camera = '/'.join([self.datasetPath, "cam_a"])

		# select some images
		imagesPaths = list(paths.list_images(camera))[:self.nImgs]
		labelsAndPaths = list(enumerate(imagesPaths))
		return(labelsAndPaths)

	def queryImgPath(self, queryNum: int=None) -> (int, str):
		camera = '/'.join([self.datasetPath, "cam_b"])
		if queryNum is None:
			queryNum = random.randint(0, self.nImgs)
		queryPath = list(paths.list_images(camera))[queryNum]
		return((queryNum, queryPath))

############################     M100 dataset functions                #########################################
class M100Dataset(Dataset):
	def __init__(self, nImgs: int=20) -> None:
		self.N_IMAGES = 80 #the number of images in the dataset (do not change)
		self.datasetPath = "./imagesIn/100m_dataset"
		self.myQueryNum = None
		self.nImgs = nImgs
		if nImgs > self.N_IMAGES-1:
			print("Warning: not enoght images in the dataset (set to", self.N_IMAGES-1, ")")
			self.nImgs = self.N_IMAGES-1

	def getLabelsAndPaths(self) -> "list of couple":
		samples = random.sample(range(self.N_IMAGES), self.nImgs+1)
		self.myQueryNum = samples[0]

		imagesPaths = list(paths.list_images(self.datasetPath))
		labelsAndPaths = []
		for s in samples[1:]:
			print("sampled name:{}, \tid:{}, \tpath:{}".format(s, s//10, imagesPaths[s]))
			labelsAndPaths.append( (s//10, imagesPaths[s]) )

		print(labelsAndPaths)
		return(labelsAndPaths)

	def queryImgPath(self, queryNum: int=None) -> (int, str):
		if queryNum is None:
			queryNum = self.myQueryNum
		queryPath = list(paths.list_images(self.datasetPath))[queryNum]
		print("\nQUERY name:{}, id:{}, path:{}".format(queryNum, queryNum//10, queryPath))
		return((queryNum//10, queryPath))


################################################################################################################
############################     Encoding Class             ####################################################
################################################################################################################
import pickle
import pprint
class DatabaseOfEncodings:
	databasePath = {
		#classifier: path of all the encodings
		#The negativePeople dataset contain 518 people/keys, and 10 images/feature vectors for each one, for a total of 5180 imgs
		"googleNet": "./encodings/googleNet_negativePeople_fullDataset.pkl",
		"resNet50":  "./encodings/resNet50_negativePeople_fullDataset.pkl"
	}

	def __init__(self, classifierType: str="resNet50", returnPath: bool=False):
		"""
			classifierType (str): which classifier to use
			returnPath (bool): if the path of the src imgs are retuned or not
		"""
		if classifierType not in self.databasePath:
			print("[INFO] Warning unknown classifier type...")
			return None
		else:
			#set main variables
			self.classifier = classifierType
			self.encodingsPath = self.databasePath[classifierType]
			self.returnPath = returnPath

			#load a dict with all the encodings of the NegativePeople dataset
			self.encodings = pickle.loads(open(self.encodingsPath, "rb").read())
			self.keys = list(self.encodings.keys())


	def getNEncodings(self, n: int=1) -> "list of list":
		""" 
			This function take n encodings from the dataset 
			(returned as a list of list, evenntually with the img source path) and remove them. 
		"""

		#each run is independent from each other (there is only a global view)
		#note: when all encodings are used this function become an empty loop
		elements = []
		paths = []
		for _ in range(n):
			#after ~5180 (518keys*10images per key) loops all will fall here
			if len(self.keys)<=0:
				print("No more elements")

			else:
				#get a random key position (lvl 1)
				indxOut = random.randint(0, len(self.keys)-1)
				#use self.keys to access an existing position (a person=out)
				elOut = self.encodings[ self.keys[indxOut] ]

				#get a random indx for the chosen person (lvl 2)
				indxIn = random.randint(0, len(elOut)-1)
				#get the image=in (feature vector) associated to the index
				elIn = elOut[indxIn]	#this is the value to return

				elements.append(elIn[0][0].tolist())
				if self.returnPath:
					paths.append(elIn[1])

				#remove the chosen image to do not pick it up twice
				#move it to the last position and then reduce the list size
				elOut[indxIn], elOut[-1] = elOut[-1], elOut[indxIn]	#easiest way to do a swap in python
				self.encodings[ self.keys[indxOut]] = elOut[:-1] #delete the extracted element (In)

				#if the list is empty now remove it also
				if len(self.encodings[ self.keys[indxOut]])==0:
					del self.encodings[ self.keys[indxOut]]	#delete from the dictionary
					self.__removeKey(indxOut)				#delete from the existing keys
		return((elements, paths))

	def __removeKey(self, i: int) -> None:
		""" Remove the element at the position i, in the self.keys list. """
		if 0<=i and i<len(self.keys):
			#first move to the last position, and second reduce the list dimension
			self.keys[i] = self.keys[-1]
			self.keys = self.keys[:-1]

		

		

