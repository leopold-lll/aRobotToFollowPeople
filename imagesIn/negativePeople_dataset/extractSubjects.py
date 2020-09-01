#The goal of this script is to extract from a dataset (firstly ideated for the Market-1501 dataset) a set of images according to the img id.
#The taken images will be part of a new dataset which can be used as negative training samples for realtime person reIdentification.
#The new dataset (NegativePeople) is composed of k[=10] folders containing the same people repeated with differend views.
#The number of images in the folders is decreasing => all the people in folder 10 are present in folder 0 but not viceversa.

#USAGE: python3 extractSubjects.py
#remove folders: rm -rf samplesSet*

# import the necessary packages
from imutils import paths
import random
import cv2
import os


def main():
	#parameters of the execution
	pathSourceDataset = "../../../../Dataset/Market-1501/bounding_box_train"	#dataset location
	pathDestiantion = "repeatedPeople/set"	#name of the output folder[s]
	numOfOccurrencies = 10			#number of output folders (each one will contain a sample for each person [if exist])

	#create the destination folders if does not exist yet
	folders = []
	for i in range(numOfOccurrencies):
		#each folder is identified by an incremental number
		folder = "".join([pathDestiantion, str(i)])
		folders.append(folder)

		if not os.path.exists(folder):
			os.mkdir(folder)

	
	#loop useful variables
	actualNum = None
	pathsOfNums = []

	#loops over all the images in the source folder
	imagePaths = list(paths.list_images(pathSourceDataset))
	imagePaths.append(imagePaths[0]) #manage a corner case to consider and add even the last person in the dataset
	for path in imagePaths:
		#take the number (aka id of the person) for the path
		imgNum = path.split("/")[-1].split("_")[0]

		if imgNum == actualNum:
			#processing an already known person number
			pathsOfNums.append(path)

		else:
			#a new person number found
			if actualNum is not None:
				#copy random images of person (identified by the number) from the dataset
				print("Moving person", actualNum, "with", min(len(pathsOfNums), numOfOccurrencies), "imgs out of", numOfOccurrencies)
				actualNum = ".".join([actualNum, "jpg"])

				#select some samples at random or take all if not enought
				imgIds = []
				if len(pathsOfNums)>numOfOccurrencies:
					imgIds = random.sample(range(0, len(pathsOfNums)), numOfOccurrencies)
				else:
					#managae the case when one person does not appear in all sub set
					imgIds = range(len(pathsOfNums))

				#effectively copy the images from source to destination dataset
				for (id, folder) in zip(imgIds, folders):
					destImgName = "/".join([folder, actualNum])
					command = " ".join(["cp", pathsOfNums[id], destImgName])
					os.system(command)
				
					#print("executed command run:", command)
				#print("\n")


			#reset the list of paths of the same person
			actualNum = imgNum
			pathsOfNums = [path]


if __name__ == "__main__":
	main()
