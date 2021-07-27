import cv2
import os
import numpy as np
import random
import time
import math

imagePath = "..//..//DOTA20DataSets//train//images"
labelPath = "..//..//DOTA20DataSets//train//labels"

imageName = "P0000.png"

#------------------------------------------------------------------------
# Function Name: drawBoundingBoxes()
# Description: This function plot bounding box on the input image and 
# can display or save the output
#------------------------------------------------------------------------
def drawBoundingBoxes(imagePath, labelPath, imageName, mode, outPath=None):
	'''
	* imagePath: Path of image folder
	* labelPath: Path of label folder
	* imageName: Name of image on which the operation is performed
	* mode     : 'show' or 'save'
	* outPath  : if mode is 'save' then output path
	''' 
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


#drawBoundingBoxes(imagePath, labelPath, imageName, mode='save', outPath="..//..//DOTA20DataSets//testOutput")

#------------------------------------------------------------------------
# Function Name: getBBoxList()
# Description: return list of all the objects coordinates in a label file
# in format [x1, y1, x2, y2, x3, y3, x4, y4]
#------------------------------------------------------------------------
def getBBoxList(filepath):
	'''
	* filepath: Path of label file
	'''
	ObjectCoordinatesList = []
	with open(filepath, 'r') as labelFile:
		ObjectList = labelFile.readlines()
	for Object in ObjectList:
		ObjectCoordinatesList.append([int(float(i)) for i in Object.split(' ')[:8]])
	return ObjectCoordinatesList

#ObjectCoordinatesList = getBBoxList(labelPath+os.sep+'P0000.txt')

#------------------------------------------------------------------------
# Function Name: polygonToRotRectangle()
# Description: Rotated Rectangle in format [cx, cy, w, h, theta]
#------------------------------------------------------------------------
def polygonToRotRectangle(bbox):
    '''
    * param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    '''
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(2,4),order='F')
    angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])

    center = [[0],[0]]

    for i in range(4):
        center[0] += bbox[0,i]
        center[1] += bbox[1,i]

    center = np.array(center,dtype=np.float32)/4.0

    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose(),bbox-center)

    xmin = np.min(normalized[0,:])
    xmax = np.max(normalized[0,:])
    ymin = np.min(normalized[1,:])
    ymax = np.max(normalized[1,:])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    return [float(center[0]),float(center[1]),w,h,angle]

#print(polygonToRotRectangle(getBBoxList(labelPath+os.sep+'P0000.txt')[0]))