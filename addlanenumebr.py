import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# img_rgb = cv.imread('mario.png')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('mario_coin.png',0)
# w, h = template.shape[::-1]
# res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
# threshold = 0.8

# cv.imwrite('res.png',img_rgb)
import numpy as np
import cv2 
import random
#backSub = cv.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()
# maskZone=[(405,340),(459,317),(471,338),(421,369)]
# maskZone = np.array(maskZone, np.int32).reshape((-1,1,2))
pointSrc=[]
zoneColor=[]
finishedInput=False

def getRandomColor():
	return (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
def displayPoint(displayImg,points,colors):
	font = cv2.FONT_HERSHEY_SIMPLEX 
	for i in range(len(pointSrc)):

			# if(finishedInput):
			#   cv2.circle(displayImg,points[i],5,colorInput[i],3)
			# else:
		cv2.circle(displayImg,points[i],5,colors,3)
		cv2.putText(displayImg, str(i), points[i], font,1, colors, 1, cv2.LINE_AA)
	zoneDawing=np.array([pointSrc], np.int32).reshape((-1,1,2))
	cv2.polylines(displayImg,[zoneDawing],True,(0,255,0),2)

	return displayImg 
def getPosition(event,x,y,flags,param):
	global mouseX,mouseY,pointSrc
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX,mouseY = x,y
		print(x,"|",y)
		if(finishedInput and len(pointInput)<=len(pointSrc)):
			print("canot add")
		else:
			tmpColor=getRandomColor()
			#colorSrc.append(tmpColor)
			pointSrc.append((mouseX,mouseY))
			print("added ")


def setZone(img):
	global pointSrc
	font = cv2.FONT_HERSHEY_SIMPLEX 
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',getPosition)
	
	zones=[]
	zoneColor=[]
	while(1):
		# print("waiting")
		displayImg=img.copy()
		displayImg=displayPoint(displayImg,pointSrc,(255,255,0))
		cv2.putText(displayImg, "you are drawing zone:"+str(len(zones)), (100,50), font,1, (0,0,255), 2, cv2.LINE_AA)
		for i in range(len(zones)):
	
			cv2.polylines(displayImg,[zones[i]],True,zoneColor[i],4)

		cv2.imshow('image',displayImg)
		k = cv2.waitKey(20) & 0xFF
		if k == 27:
			break
		elif k == ord('d'):
			if(len(pointSrc)>0):
				pointSrc=pointSrc[:-1]
				print("remove last ",len(pointSrc))
		elif k == ord('c'):
			if(len(zones)>0):
				zones=zones[:-1]
				zoneColor=zoneColor[:-1]
				print("remove last zone ",len(pointSrc))
		elif k == ord('a'):
			zones.append(np.array([pointSrc], np.int32).reshape((-1,1,2)))
			zoneColor.append(getRandomColor())
			pointSrc=[] 
	return zones




def resetZone(img,zones):
	global pointSrc
	font = cv2.FONT_HERSHEY_SIMPLEX 
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',getPosition)
	

	while(1):
		# print("waiting")
		displayImg=img.copy()
		displayImg=displayPoint(displayImg,pointSrc,(255,255,0))
		cv2.putText(displayImg, "you are drawing zone:"+str(len(zones)), (100,50), font,1, (0,0,255), 2, cv2.LINE_AA)
		for i in range(len(zones)):
	
			cv2.polylines(displayImg,[zones[i]],True,(255,255,255),4)

		cv2.imshow('image',displayImg)
		k = cv2.waitKey(20) & 0xFF
		if k == 27:
			break
		elif k == ord('d'):
			if(len(pointSrc)>0):
				pointSrc=pointSrc[:-1]
				print("remove last ",len(pointSrc))
		elif k == ord('c'):
			if(len(zones)>0):
				zones=zones[:-1]
				#zoneColor=zoneColor[:-1]
				print("remove last zone ",len(pointSrc))
		elif k == ord('a'):
			zones.append(np.array([pointSrc], np.int32).reshape((-1,1,2)))
			#zoneColor.append(getRandomColor())
			pointSrc=[] 
	return zones





def centroid(vertexes):

	x_list=[]
	y_list=[]
	for i in range(len(vertexes)):
		print(vertexes[i][0][0])
		print(vertexes[i][0][1])


		x_list.append(vertexes[i][0][0])
		y_list.append(vertexes[i][0][1])
	 # _x_list = [vertex [0] for vertex in vertexes]
	 # _y_list = [vertex [1] for vertex in vertexes]
	 # _len = len(vertexes)
	x = int(sum(x_list) / len(vertexes))
	y = int(sum(y_list) / len(vertexes))
	print((x,y))

	return (x,y)




pointSrc=[]
filename="DJI_20250408122911_0004stabilizationstabCheckedoutzones.npy"
filePath="./output_stable_3/"+filename
cap = cv.VideoCapture(filePath)

cv2.namedWindow('image')
cv2.setMouseCallback('image',getPosition)




scale=1
videoWidth=cap.get(3)
videoHeight=cap.get(4)
videoFps=cap.get(5)
ret, img = cap.read()
zones=setZone(img)
np.save("./"+filename[:-4]+"zones.npy",zones,allow_pickle=True)
#zones=np.load("./"+filename[:-4]+"zones.npy",allow_pickle=True)
# zones=resetZone(img,zones.tolist())
# np.save("./"+filename[:-4]+"zones.npy",zones,allow_pickle=True)
#zpme
blank_image = np.zeros((round(videoHeight*scale),round(videoWidth*scale),3), np.uint8)
# for zone in zones:
# # maskZone=[(405,340),(459,317),(471,338),(421,369)]
#     maskZone = np.array(zone, np.int32).reshape((-1,1,2))
#     blank_image=cv2.fillPoly(blank_image,[maskZone],(255,255,255))
grey_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
grey_image = np.uint8(grey_image)
randomCollor=[]
for i in range(len(zones)):
	randomCollor.append(getRandomColor())


imgINIT = cv2.imread(filename, cv2.IMREAD_COLOR)
while True:
	img=imgINIT.copy()
	for i in range(len(zones)):
		# if(i < 39 or i>42 ):
		# 	continue 
		cv2.polylines(img,[zones[i]],True,randomCollor[i],5)
		#cv2.polylines(blank_image2,[zones[i]],True,randomCollor[i],5)
		centerPoint=centroid(zones[i])
		centerPoint=(centerPoint[0],centerPoint[1]+20)
		#cv2.putText(blank_image2,str(i),centerPoint,cv2.FONT_HERSHEY_SIMPLEX,1,randomCollor[i],2)
		cv2.putText(img,str(i),centerPoint,cv2.FONT_HERSHEY_SIMPLEX,3,randomCollor[i],5)

		
	cv2.imshow("for2",img)
	#cv2.imshow("blank_image",blank_image2)
	k = cv2.waitKey(20) & 0xFF
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
