# The goal of this file is to load images from the negative people dataset and create all their encodings
# with resNet50 or googleNet. The created encodings will be used in the main flow of the code to fit KNN with
# not only positive samples but also negative once.
# The second part of code is useful for test how knn works on the generated encodings.

# USAGE: 
# python createPerson_encodings.py --encOut encodings --createEncoding
# python createPerson_encodings.py --encOut encodings --createEncoding --model resNet50 --imgOut imagesOut/s5n100_ -s 5 -n 100
# python createPerson_encodings.py --encOut encodings --createEncoding --model googleNet --imgOut imagesOut/s9n90_ -s 9 -n 90

# python createPerson_encodings.py --encOut encodings --model resNet50 --imgOut imgsOut/s5n100_ -s 5 -n 100
# python createPerson_encodings.py --encOut encodings --model googleNet --imgOut imagesOut/s9n90_ -s 9 -n 90

# import the necessary packages
from scipy import spatial
import argparse
import pickle
import heapq
import cv2
import pprint

#import custom functions
import sys
import os
sys.path.append(os.path.abspath('./utils'))
from dataset import NegativePeopleDataset
from model import ResNet50, GoogleNet
from gridView import createGrid
import pprint

############################     Flow controll functions     ###################################################
def computeEncoding(model, imagePath):
	""" Given a single image path, compute its encoding. """
	# load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	# compute the embedding
	encoding = model.feed(image)
	return(encoding)

def encodeDataset(model, dataset, encodingsFile) -> dict:
	"""	Scan a part of the dataset and fill the N-dimensional space with the input samples. """
	print("[INFO] Quantifying samples...")

	# grab the paths to the input samples in our dataset
	labelsAndPaths = dataset.getLabelsAndPaths()
	# initialize the list of known encodings
	encodings = {}

	# loop over the image paths
	for (i, (l, path)) in enumerate(labelsAndPaths):
		print("[INFO] processing image {}/{}".format(i + 1, len(labelsAndPaths)))
		if l not in encodings:
			# add to the dictionary a new key
			encodings[l] = [[computeEncoding(model, path), path]]
		else:
			encodings[l].append([computeEncoding(model, path), path])
		
		#print(len(encodings[l][0][0])) #1024 for googleNet and 2048 for resNet50

	# dump the people encodings to disk
	print("\n[INFO] Serializing encodings...")
	f = open(encodingsFile, "wb")
	f.write(pickle.dumps(encodings))
	f.close()

	#return the dicotionary
	return(encodings)

def pairPeople(model, dataset, encodings: dict, queryNum: int=None, imgOut: str=None) -> None:
	""" Given a new image try to match it with the encodings previously calculated. """
	print("[INFO] Try pairing people...")

	(queryLabel, queryPath) = dataset.queryImgPath(queryNum)	# get a random query

	if queryPath is not None:
		queryEncode = computeEncoding(model, queryPath)				# compute its encoding
		print("compute the query for the class id:", queryLabel)

		labels = []
		paths = []
		distances = []
		for (label, listEnc) in encodings.items():
			# each key can store multiple points (AKA each identity can has multiple samples pictures)
			for (enc, path) in listEnc:
				labels.append(label)
				paths.append(path)
				dist = spatial.distance.euclidean(queryEncode, enc)
				distances.append(dist)
				#print("distance from B{} and A{} is: {:.3f}".format(imgB, k, dist))

		topK = heapq.nsmallest(9, zip(distances, labels, paths))
		for (i, (score, id, pth)) in enumerate(topK):
			print("The {}° match has score {:.3f} and id {}".format(i+1, score, id))
		print()

		#create the Grid for a better visualization
		if imgOut is not None: #aka: I want to see the result
			query = [(queryLabel, queryPath)]
			# remove useless distance value and add the query img
			bestMatch = query + [el[1:] for el in topK]

			#create a grid visualization and show it
			gridImg = createGrid(bestMatch, 3, 3)
			if gridImg is not None:
				name="".join([imgOut, str(queryLabel), ".jpg"])
				cv2.imwrite(name, gridImg)

				cv2.imshow("grid", gridImg)
				cv2.waitKey(-1)
				cv2.destroyAllWindows()
	

############################     Main functions     ############################################################

def parseArguments() -> None:
	"""Construct the argument parser and parse the arguments."""
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model",		type=str, default="resNet50",		help="A model name, choises: resNet50, googleNet")
	ap.add_argument("-e", "--encOut",		type=str, default="encsOut",help="Path to the output directory for the encoding.")
	ap.add_argument("-o", "--imgOut",		type=str, default="imgsOut",help="Path to the output directory for the images.")
	ap.add_argument("-c", "--createEncoding",		action="store_true", default=False,	help="If create the encoding of the dataset or load it.")
	ap.add_argument("-n", "--nImgs",		type=int, default=10,		help="The number of selected people for training.")
	ap.add_argument("-s", "--sets",			type=int, default=5,		help="If NegativePeopleDataset, how many sets of imges will be used for training (min 1 and max 9). ")

	args = vars(ap.parse_args())
	return(args)

def main() -> None:
	args = parseArguments()

	#load the dataset class
	print("loading NegativePeopleDataset.")
	dataset = NegativePeopleDataset(args["nImgs"], nTestDir=args["sets"])

	#create model
	model = None
	if	 args["model"]=="resNet50":
		model = ResNet50()
	elif args["model"]=="googleNet":
		model = GoogleNet()

	if dataset is not None and model is not None:
		#create or load the encodings dictionary
		encodingsFile = ''.join([args["encOut"], "/", args["model"], "_negativePeople_s", str(args["sets"]), "-n", str(args["nImgs"]), ".pkl"])
		if args["createEncoding"]:
			print("[INFO] Creating encodings...")
			encodings = encodeDataset(model, dataset, encodingsFile)
		else:
			print("[INFO] Loading encodings...")
			encodings = pickle.loads(open(encodingsFile, "rb").read())

		#pairing people to the space of encodings
		while True:
			q=0
			i = input("\nquery number (error will quit):")
			if i!="":
				try:
					q = int(i)
				except:
					print("The input is not a number...")
					break
			imgBaseName = ''.join([args["imgOut"], args["model"], "_negativePeople_"])
			pairPeople(model, dataset, encodings, q, imgBaseName)

if __name__ == "__main__":
	main()
	

