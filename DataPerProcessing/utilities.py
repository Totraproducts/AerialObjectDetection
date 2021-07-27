import cv2
import os
import numpy as np
import random
import time

imagePath = "..//..//DOTA20DataSets//train//images"
labelPath = "..//..//DOTA20DataSets//train//labels"

imageName = "P0000.png"

def drawBoundingBoxes(imagePath, labelPath, imageName, mode, outPath=None):
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.4
	fontThickness = 1
	boundingBoxThickness = 1
	isClosed = True

	#imageNameList = os.listdir(imagePath)
	image = cv2.imread(imagePath + os.sep + imageName)
	with open(labelPath + os.sep + imageName.split('.')[0]+'.txt', 'r') as labelFile:
		ObjectList = labelFile.readlines()

		for Object in ObjectList:
			ObjectCoordinate = [int(float(i)) for i in Object.split(' ')[:8]]
			ObjectName = Object.split(' ')[8]
			pts = np.array([[ObjectCoordinate[0],ObjectCoordinate[1]],
				[ObjectCoordinate[2],ObjectCoordinate[3]],
				[ObjectCoordinate[4],ObjectCoordinate[5]],
				[ObjectCoordinate[6],ObjectCoordinate[7]]],
				np.int32)
			color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))

			image = cv2.polylines(image, [pts], isClosed, color, boundingBoxThickness)
			image = cv2.putText(image, ObjectName, (ObjectCoordinate[0],ObjectCoordinate[1]), font, fontScale, color, fontThickness)

	imageRes = image.shape
	image = cv2.resize(image, (imageRes[1]//1, imageRes[0]//1))
	if mode == 'show':
		cv2.imshow('Image',image)
		cv2.waitKey()
	elif mode == 'save':
		if outPath is not None:
			cv2.imwrite(outPath+os.sep+imageName.split('.')[0]+'_labeled.png', image)
		else:
			print('No Output path Provided')


drawBoundingBoxes(imagePath, labelPath, imageName, mode='save', outPath="..//..//DOTA20DataSets//testOutput")
