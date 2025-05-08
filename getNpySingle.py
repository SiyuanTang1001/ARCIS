# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from skimage import exposure
time =0
thresh = 1
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
fileName="./S1000125stabChecked.mp4"		
capA = cv2.VideoCapture(fileName)
capB = cv2.VideoCapture("S1000123bg.png")
capA.set(cv2.CAP_PROP_POS_FRAMES, 16)
ret, frameA = capA.read()
ret, frameB = capB.read()
blank_image = frameB
mouseX,mouseY=0,0
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('imageA')
cv2.setMouseCallback('imageA',getPosition)
points=[]
pointA2B=[]
pointB2A=[]
cv2.namedWindow('imageB')
cv2.setMouseCallback('imageB',getPosition)
scale=0.5
while(1):
	frameADisplay=frameA.copy()
	pointCounter=0
	for i in range(len(points)):
		cv2.circle(frameADisplay,points[i],4,(255,0,0),1)
		cv2.putText(frameADisplay, str(i), points[i], font,1, (255, 0, 0), 2)
	cv2.imshow('imageA',frameADisplay)
	frameBDisplay=frameB.copy()
	frameBDisplay=cv2.resize(frameBDisplay,(0,0),fx=scale,fy=scale)
	cv2.imshow('refer',frameBDisplay)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
pointA2B=points
points=[]
cv2.destroyAllWindows()
cv2.namedWindow('imageB')
cv2.setMouseCallback('imageB',getPosition)
while(1):
	frameADisplay=frameA.copy()
	frameBDisplay=frameB.copy()
	for i in range(len(pointA2B)):
		cv2.circle(frameADisplay,pointA2B[i],2,(255,0,0),1)
		cv2.putText(frameADisplay, str(i), pointA2B[i], font,2, (255, 0, 0), 2)
	for i in range(len(points)):
		cv2.circle(frameBDisplay,points[i],4,(255,0,0),1)
		cv2.putText(frameBDisplay, str(i), points[i], font,1, (255, 0, 0), 2)
	cv2.imshow('imageB',frameBDisplay)
	frameADisplay=cv2.resize(frameADisplay,(0,0),fx=scale,fy=scale)
	cv2.imshow('refer',frameADisplay)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
pointB2A=points
points=[]

src = np.array(pointA2B,np.float32)
dst = np.array(pointB2A,np.float32)
#h2=np.load("out2.npy")
h,status= cv2.findHomography(src,dst)
np.save(fileName[:-4]+".npy",h)
result = cv2.warpPerspective(frameA, h,(frameB.shape[1],frameA.shape[0]))

cv2.imshow('result',result)
# cv2.imshow('result1',result)
cv2.waitKey(0)
