import numpy as np
import cv2 as cv
import csv
import time
def isLightOn(frame,region):
    imCrop2 = resized[int(region[1]):int(region[1]+region[3]), int(region[0]):int(region[0]+region[2])]
    gray = cv.cvtColor(imCrop2, cv.COLOR_RGB2HSV)
    lower_red = np.array([0,70,153])
    upper_red = np.array([255,255,255])
    mask1 = cv.inRange(gray, lower_red, upper_red)
        #red_ratio=(cv.countNonZero(resized))/(resized.size/3)
    font = cv.FONT_HERSHEY_SIMPLEX
    light=False
    if(cv.countNonZero(mask1)>600):
            #counter=0
            #lightWork=True
        return True
            #cv.putText(frame, "light On", (1000,500), font,5, (255, 255, 0), 5)
    else:
            #counter+=1
        return False


fileName="./NW_1130_1400.mp4"
cap = cv.VideoCapture(fileName)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
#cap.set(1,2500)
ret, frame = cap.read()
r = cv.selectROI(frame)
imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
print(r)





resized = cv.resize(imCrop, None, fx=4, fy=4, interpolation=cv.INTER_AREA)
r2 = cv.selectROI(resized)
r3 = cv.selectROI(resized)
r4 = cv.selectROI(resized)
print(r2)
print(r3)
print(r4)
#imCrop2 = resized[int(r2[1]):int(r2[1]+r2[3]), int(r2[0]):int(r2[0]+r2[2])]
#backSub = cv.createBackgroundSubtractorMOG2()
#r=(1524, 265, 58, 27)
#r2=(44, 40, 162, 35)
counter=0
lightWork=False
start = time.time()
with open(fileName[:-4]+"light.csv", mode='w') as light_file:
    light_file = csv.writer(light_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    light_file.writerow(['frame', 'leftLight', 'middleLight','rightLight'])


frameCounter=0
timeSum=0
ret, frame = cap.read()
totalFrame=cap.get(cv.CAP_PROP_FRAME_COUNT)
cap = cv.VideoCapture(fileName)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    resized = cv.resize(imCrop, None, fx=4, fy=4, interpolation=cv.INTER_AREA)
    light1status=isLightOn(resized,r2)
    light2status=isLightOn(resized,r3)
    light3status=isLightOn(resized,r4)
    cv.putText(frame, "l1:"+str(light1status), (100,350), cv.FONT_HERSHEY_SIMPLEX,5, (255, 0, 0), 5)
    cv.putText(frame, "l2:"+str(light2status), (100,450), cv.FONT_HERSHEY_SIMPLEX,5, (255, 0, 0), 5)
    cv.putText(frame, "l3:"+str(light3status), (100,550), cv.FONT_HERSHEY_SIMPLEX,5, (255, 0, 0), 5)
    with open(fileName[:-4]+"light.csv", mode='a') as light_file:
        light_file = csv.writer(light_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        light_file.writerow([frameCounter, light1status, light2status,light3status])

        frameCounter+=1


    end = time.time()
    timeCost=end - start
    timeSum+=timeCost
    start = time.time()
    if(frameCounter>0):
        avgTime=timeSum/frameCounter
        avgFps=1/avgTime

        print(round(frameCounter/totalFrame*100,3)," %|avgFPS",round(avgFps,1),"procresstime",round((totalFrame-frameCounter)/avgFps/60,2))




    
    #fgMask = backSub.apply(imCrop)
    #cv.imshow("frame4",frame)
    # cv.imshow('frame1', gray)
    # cv.imshow('frame2', imCrop2)
    # cv.imshow('frame3', mask1)
    
    #if cv.waitKey(1) == ord('q'):
        #break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
