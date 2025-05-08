import numpy as np
import cv2 as cv
import csv
import time
cap = cv.VideoCapture("./NW_1130_1400.mp4")
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
print(r2)
#imCrop2 = resized[int(r2[1]):int(r2[1]+r2[3]), int(r2[0]):int(r2[0]+r2[2])]
#backSub = cv.createBackgroundSubtractorMOG2()
#r=(1524, 265, 58, 27)
#r2=(44, 40, 162, 35)
counter=0
lightWork=False
start = time.time()
with open('NW_1130_1400light.csv', mode='w') as light_file:
    light_file = csv.writer(light_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    light_file.writerow(['frame', 'light', 'rrfb'])


frameCounter=0
timeSum=0
ret, frame = cap.read()
totalFrame=cap.get(cv.CAP_PROP_FRAME_COUNT)
cap = cv.VideoCapture("NW_1130_1400.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #desired_dim = (30, 90) # width, height
    #frame = cv.resize(np.array(frame), desired_dim, interpolation=cv.INTER_LINEAR)
    
    imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    resized = cv.resize(imCrop, None, fx=4, fy=4, interpolation=cv.INTER_AREA)
    imCrop2 = resized[int(r2[1]):int(r2[1]+r2[3]), int(r2[0]):int(r2[0]+r2[2])]

    gray = cv.cvtColor(imCrop2, cv.COLOR_RGB2HSV)

    # lower mask (0-10)
    #lower_red = np.array([0,70,50])
    #upper_red = np.array([100,255,255])
    #mask0 = cv.inRange(gray, lower_red, upper_red)
    # lower_red = np.array([0,120,70])
    # upper_red = np.array([10,255,255])
    # mask0 = cv.inRange(gray, lower_red, upper_red)

    # upper mask (170-180)
    #resized = cv.resize(imCrop, None, fx=4, fy=4, interpolation=cv.INTER_AREA)
    #30,43,8,88.6
    #42,57,79
    lower_red = np.array([0,70,153])
    upper_red = np.array([255,255,255])
    mask1 = cv.inRange(gray, lower_red, upper_red)
    #red_ratio=(cv.countNonZero(resized))/(resized.size/3)
    font = cv.FONT_HERSHEY_SIMPLEX
    light=False
    if(cv.countNonZero(mask1)>600):
        counter=0
        lightWork=True
        light=True
        cv.putText(frame, "light On", (1000,500), font,5, (255, 255, 0), 5)
    else:
        counter+=1
        light=False
        cv.putText(frame, "light Off", (1000,500), font,5, (255, 0, 0), 5)

    if(counter>7):
        lightWork=False
    if(lightWork):
        cv.putText(frame, "rrfb  On", (1000,650), font,5, (255, 255, 0), 5)
    else:

        cv.putText(frame, "rrfb  Off", (1000,650), font,5, (255, 0, 0), 5)
    with open('NW_1130_1400.mp4', mode='a') as light_file:
        light_file = csv.writer(light_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        light_file.writerow([frameCounter, light, lightWork])

        frameCounter+=1



    end = time.time()
    timeCost=end - start
    timeSum+=timeCost

    start = time.time()

    if(frameCounter>0):
        avgTime=timeSum/frameCounter

        avgFps=1/avgTime
        # avgTimePPfp
        # print(1/avgTime)
    print(round(frameCounter/totalFrame*100,3)," %|avgFPS",round(avgFps,1),"procresstime",round((totalFrame-frameCounter)/avgFps/60,2))




    
    #fgMask = backSub.apply(imCrop)
    # cv.imshow("frame4",frame)
    # cv.imshow('frame1', gray)
    # cv.imshow('frame2', imCrop2)
    # cv.imshow('frame3', mask1)
    
    # if cv.waitKey(1) == ord('q'):
    #     break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
