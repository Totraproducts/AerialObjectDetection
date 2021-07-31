import cv2
import os
import numpy as np
import random
import time
import math

imagePath = "..//..//DOTA20DataSets//train//images"
labelPath = "..//..//DOTA20DataSets//train//labels"

imageName = "P0001.png"
labelName = "P0001.txt"


#------------------------------------------------------------------------
# Function Name: printProgressBar()
# Description: Call in a loop to create terminal progress bar
#------------------------------------------------------------------------
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    '''
    * iteration   - Required  : current iteration (Int)
    * total       - Required  : total iterations (Int)
    * prefix      - Optional  : prefix string (Str)
    * suffix      - Optional  : suffix string (Str)
    * decimals    - Optional  : positive number of decimals in percent complete (Int)
    * length      - Optional  : character length of bar (Int)
    * fill        - Optional  : bar fill character (Str)
    * printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


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
	ObjectNamesList = []
	ObjectDifficultyList = []
	with open(filepath, 'r') as labelFile:
		ObjectList = labelFile.readlines()
	for Object in ObjectList:
		splitline = Object.split(' ')
		ObjectCoordinatesList.append([int(float(i)) for i in splitline[:8]])
		ObjectNamesList.append(splitline[8])
		if len(splitline) > 9:
			ObjectDifficultyList.append(splitline[9])
		else:
			ObjectDifficultyList.append('0')
	return ObjectCoordinatesList, ObjectNamesList, ObjectDifficultyList

#ObjectCoordinatesList, ObjectNamesList = getBBoxList(labelPath+os.sep+'P0000.txt')

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

#------------------------------------------------------------------------
# Function Name: cropImageBBox()
# Description: Crop the object and make bg black
#------------------------------------------------------------------------
def cropImageBBox(imagePath, bbox, mode='show', outPath=None):
	'''
	* imagePath: raw image path from data set
	* bbox     : The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
	* mode     : 'show' or 'save'
	* outPath  : if mode is 'save' then output path
	'''
	x1, y1, x2, y2, x3, y3, x4, y4 = bbox
	image = cv2.imread(imagePath)
	mask = np.zeros(image.shape, dtype=np.uint8)
	# mask defaulting to black for 3-channel and transparent for 4-channel
	# (of course replace corners with yours)
	roi_corners = np.array([[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]], dtype=np.int32)
	# fill the ROI so it doesn't get wiped out when the mask is applied
	channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
	ignore_mask_color = (255,)*channel_count
	cv2.fillPoly(mask, roi_corners, ignore_mask_color)
	# from Masterfool: use cv2.fillConvexPoly if you know it's convex

	# apply the mask
	masked_image = cv2.bitwise_and(image, mask)
	xmin = min(x1,x2,x3,x4)
	xmax = max(x1,x2,x3,x4)
	ymin = min(y1,y2,y3,y4)
	ymax = max(y1,y2,y3,y4)
	image = masked_image[ymin:ymax, xmin:xmax] 
	# save the result
	if mode == 'show':
		cv2.imshow('Image',image)
		cv2.waitKey()
	elif mode == 'save':
		if outPath is not None:
			cv2.imwrite(outPath, image)
		else:
			print('No Output path Provided')

#------------------------------------------------------------------------
# Function Name: sendToCropImageBBox()
# Description: Preprocessing before sending to cropImageBBox
#------------------------------------------------------------------------
def sendToCropImageBBox(mainPath):
	'''
	* mainPath: Path where sub-folder with object class name will create
	'''
	ObjectCoordinatesList, ObjectNamesList, ObjectDifficultyList = getBBoxList(labelPath+os.sep+labelName)
	print('Number of Objects: {}'.format(len(ObjectCoordinatesList)))
	for dirName in set(ObjectNamesList):
		if not os.path.exists(mainPath+os.sep+dirName):
	    			os.makedirs(mainPath+os.sep+dirName)
	printProgressBar(0, len(ObjectCoordinatesList), prefix = 'Progress:', suffix = 'Complete', length = 80)
	for i in range(len(ObjectCoordinatesList)):
		printProgressBar(i + 1, len(ObjectCoordinatesList), prefix = 'Progress:', suffix = 'Complete', length = 80)
		path = mainPath+os.sep+ObjectNamesList[i]+os.sep+ObjectNamesList[i]+'_'+str(i)+'.png'
		cropImageBBox(imagePath+os.sep+imageName, ObjectCoordinatesList[i], 'save', path)

#sendToCropImageBBox('Objects')