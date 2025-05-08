# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
from skimage import exposure
# time =0
# thresh = 1
def getPosition(event,x,y,flags,param):
	global mouseX,mouseY,pointA2B,points
	# print("event : ",event)
	# print(x,y)
	if event == 1:
		mouseX,mouseY = x,y
		points.append([x,y])
		#cv2.circle(frameB,(x,y),10,(255,0,0),-1)
		print("left",x,y)
	if event == 2:
		mouseX,mouseY = x,y
		points=points[:-1]
		#cv2.circle(frameB,(x,y),10,(255,255,0),-1)
		print("rigth",x,y)




fileName="S1000125stabChecked.mp4"
npyName=fileName[:-4]+".npy"
outputName="./output/"+fileName[:-4]+"aliened.mp4"
filename=fileName
h=np.load(npyName)

capA = cv2.VideoCapture(fileName)
# capB = cv2.VideoCapture("./tivoliBaseMap.png")

frameCounter=0
videoRate=1
videoWidth=capA.get(3)
videoHeight=capA.get(4)
videoFps=capA.get(5)
totalFrame=capA.get(cv2.CAP_PROP_FRAME_COUNT)
vid_writerCap = cv2.VideoWriter(outputName, cv2.VideoWriter_fourcc('M','P','4','V'), videoFps, (int(videoWidth),int(videoHeight)))

timeSum=0
# ret2, frameB = capB.read()
# frameB=cv2.resize(frameB, None, fx=videoRate, fy=videoRate, interpolation=cv2.INTER_AREA)

while True:
	start = time.time()
	frameCounter+=1
	ret1, frameA = capA.read()

	if not ret1:
		print("Can't receive frame (stream end?). Exiting ...")
		break

	# blank_image = frameB.copy()

	result2 = cv2.warpPerspective(frameA, h,(frameA.shape[1],frameA.shape[0]))
	vid_writerCap.write(result2)
	result2=cv2.resize(result2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	cv2.imshow('frame3', result2)

	#cv2.imshow('frameC', overlap)
	# cv2.imshow('frameCq', result)
	end = time.time()
	timeCost=end - start
	timeSum+=timeCost
	start = time.time()
	#if(frameCounter>0):
	avgTime=timeSum/frameCounter
	avgFps=1/avgTime
	print(round(frameCounter/totalFrame*100,3)," %|avgFPS",round(avgFps,1),"procresstime",round((totalFrame-frameCounter)/avgFps/60,2))
	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
