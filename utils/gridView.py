# import the necessary packages
from imutils import resize
import numpy as np
import cv2

############################   Utils   #########################################################################
def fill(image, width=None, height=None) -> "img":
	""" 
		The function take a picture and rescale it to fit the given dimensions. The rescaling is on one dimension.
		The other dimension is filled to fit the dimensions but to do not modify the image ratio.
	"""
	#wrapper function because the imutils.resize do not manage both dimension to be not None
	(hOld, wOld) = image.shape[:2]
	if width is not None and height is not None:
		ratioW = width / wOld
		ratioH = height/ hOld

		# the dimension that will resize the image is the the one with the smaller ratio (no cropping needed)
		if ratioW < ratioH:
			image = resize(image, width=width,  height=None)
		else:
			image = resize(image, width=None, height=height)

		# add an extra empty are to satisfy the reuested sizes
		(hNew, wNew) = image.shape[:2]
		image = cv2.copyMakeBorder(image, 0, height-hNew, 0, width-wNew, cv2.BORDER_CONSTANT)

	else:
		#if at least a dimension is None do not care
		image = resize(image, width=width, height=height)

	return(image)

def drawBorderAndPos(image, type: int, position: str="") -> "image":
	""" 
		Add a border to the given image. 
		The color is -1=red (aka wrong prediction), 0=blue (aka query), 1=green (aka correct prediction)
		The position is a number written at the top left corner.
	"""
	(h, w) = image.shape[:2]
	#NB: openCV works with BGR color scheme
	if type==1:		#correct=green
		color = (0, 255, 0)
	elif type==0:	#query=blue
		color = (255, 0, 0)
	else:			#wrong=red
		color = (0, 0, 255)
	cv2.rectangle(image, (0, 0), (w, h), color, 4)

	position = "" if position=="0" else position
	cv2.putText(image, position, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return(image)

def assembleGrid(images: "list of images", x: int, y: int) -> "image":
	""" 
		Given a list of n images and 2 dimensions x and y (where n=x*y), 
		the function return an image that is a gird x*y of the n images. 
	"""
	#A reference guide can be foud at: https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
	rows = []
	for i in range(y):
		row = cv2.hconcat(images[i*x : (i+1)*x])
		rows.append(row)
	grid = cv2.vconcat(rows)

	return(grid)

def createGrid(topK: "list of tuples", x: int, y: int, resize=False) -> "image":
	"""  
		Given a list of n tuples (label, path of an image) where the first tuple represent the query image.
		X and y where (where n=x*y) create the representation grid.
		Resize can be set if the input images does not have all the same shape.
	"""
	queryLabel = topK[0][0]
	imgs = []

	hMax = 0
	wMax = 0
	for (pos, (label, path)) in enumerate(topK):
		#for each image: load and draw border
		img = cv2.imread(path)
		img = drawBorderAndPos(img, 1 if label==queryLabel else -1, str(pos))
		imgs.append(img)
		
		#compute the biggest dimensions
		if resize:
			#this part is useless with Market-1501 because all the images as the same shape (64x128)
			(h, w) = img.shape[:2]
			hMax = max(hMax, h)
			wMax = max(wMax, w)

	imgs[0] = drawBorderAndPos(imgs[0], 0, "") #color back again the query img

	if resize:
		filledImgs = []
		for img in imgs:
			print(img.shape[:2])
			filledImgs.append( fill(img, wMax, hMax) )
		imgs = filledImgs
	
	#effectively compose the grid
	return( assembleGrid(imgs, x, y) )