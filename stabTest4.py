import numpy as np
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time

import math





def setZone(img):
    global pointSrc,scale
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.namedWindow('imageB')

    cv2.setMouseCallback('imageB',getPosition)
    
    zones=[]
    zoneColor=[]
    while(1):
        # print("waiting")
        displayImg=img.copy()
        displayImg=displayPoint(displayImg,pointSrc,(255,255,0))
        cv2.putText(displayImg, "you are drawing zone:"+str(len(zones)), (100,50), font,1, (0,0,255), 2, cv2.LINE_AA)
        for i in range(len(zones)):
    
            cv2.polylines(displayImg,[zones[i]],True,zoneColor[i],-1)
        displayImg=cv2.resize(displayImg,(0,0),fx=scale,fy=scale)
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
    return zones[0]



pointSrc=[]
zoneColor=[]
finishedInput=False
points=[]
import random
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





scale=1

videoName="Drone2- 16-9-50 -- 16-25-58_cropstabilization.mp4"
videoPath = "./output_stable_1/"+videoName
outputfile="./output_stable_2/"+videoName[:-4]+"bg.mp4"
outputfile2="./output_stable_2/"+videoName[:-4]+"stab.mp4"
outputImage="./output_stable_2/"+videoName[:-4]+"bg.png"
# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)
totalFrame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
videoWidth=cap.get(3)
videoHeight=cap.get(4)
videoFps=cap.get(5)
vid_writer= cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc('M','P','4','V'), videoFps, (int(videoWidth),int(videoHeight)))
# ====creat base map
success, FirstFrame = cap.read()
cap.set(0,int(totalFrame-1))
success, LastFrame = cap.read()

# kp2, des2 = orb.detectAndCompute(LastFrame,None)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
counter=0

try:
    boundZone=np.load("./output_stable/"+videoName[:-4]+"BoundZone.npy",allow_pickle=True)
except:
    cv2.namedWindow('imageA')
    cv2.setMouseCallback('imageA',getPosition)
    while(1):
        frameADisplay=LastFrame.copy()
        pointCounter=0
        for i in range(len(points)):
            cv2.circle(frameADisplay,points[i],4,(255,0,0),1)

            cv2.putText(frameADisplay, str(i), points[i], cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
        if(len(points)>3):
            tmpZOne=np.array([points], np.int32).reshape((-1,1,2))
            cv2.fillPoly(frameADisplay,[tmpZOne],(255,225,0))
        cv2.imshow('imageA',frameADisplay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    boundZone=np.array([points], np.int32).reshape((-1,1,2))
    # boundZone=setZone(FirstFrame)
    np.save("./output_stable_2/"+videoName[:-4]+"BoundZone.npy",boundZone,allow_pickle=True)

mask= np.zeros_like(LastFrame)
cv2.fillPoly(mask, pts=[boundZone], color=(255, 255, 255))
LastFrameMasked = cv2.bitwise_and(LastFrame, mask)




sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(LastFrameMasked,None)
# kp2, des2 = orb.detectAndCompute(LastFrame,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
MIN_MATCH_COUNT=4
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
counter=0




cv2.imshow("mask", LastFrameMasked)
    
counter+=1
  # quit on ESC button
cv2.waitKey(0)


while True:
    print("=================")
    if counter>=4000:
        break
    
        # Read a new frame
    ok, frame = cap.read()
#     if(counter%10!=0):
#         continue
    if not ok:
         break
        # Start timer
    timer = cv2.getTickCount()

    if(counter%30==0 or counter==0):
        kp2, des2 = sift.detectAndCompute(frame,None)
        matches = flann.knnMatch(des1,des2,k=2)
        #matches = sorted(matches, key = lambda x:x.distance)
        p1, p2 = [], []
        sumDist=0
        distCounter=0
        goodMatches = []
        minRatio = 0.7
        for m,n in matches:
            print(m.distance)
            print(n.distance)
            if(m.distance>120 or n.distance>120):
                continue
            if m.distance / n.distance < minRatio:   
                goodMatches.append(m)   
        print(len(goodMatches))
        if len(goodMatches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1,2)
            # H,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
            distSum=0
            for i in range(len(src_pts)):
                point1=(int(src_pts[i][0]),int(src_pts[i][1]))
                point2=(int(dst_pts[i][0]),int(dst_pts[i][1]))
                #distPoint=round(math.dist(point1, point2),2)
                distPoint = round(math.hypot(point2[0] - point1[0], point2[1] - point1[1]), 2)
                distSum+=distPoint
              
            avgDist=distSum/len(src_pts)
            for i in range(len(src_pts)):
                point1=(int(src_pts[i][0]),int(src_pts[i][1]))
                point2=(int(dst_pts[i][0]),int(dst_pts[i][1]))
                #distPoint=round(math.dist(point1, point2),2)
                distPoint = round(math.hypot(point2[0] - point1[0], point2[1] - point1[1]), 2)

                if(src_pts[i][0]>4500 or dst_pts[i][0] >4500):
                    continue
                if(distPoint>avgDist):
                    continue
                p1.append(point1)
                p2.append(point2)
                #frame = cv2.putText(frame, str(distPoint), point1, cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 2, cv2.LINE_AA)

                # frame = cv2.circle(frame, point1, 8, (255,0,0), 2)
                # frame = cv2.circle(frame, point2, 8, (0,0,255), 2)
                # frame = cv2.line(frame, point1,point2, (0,255,0), 6)


            H, _ = cv2.findHomography(np.float32(p2), np.float32(p1), cv2.RHO)

   
        output_frame = cv2.warpPerspective(frame, H, (frame.shape[1],frame.shape[0]))
        vid_writer.write(output_frame)
        cv2.imshow("output", output_frame)



    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    print(str(counter)+"/"+str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    
    

    #frame = cv2.resize(frame,None,fx=0.25,fy=0.25)
    
    
    counter+=1
  # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
cap.release()

vid_writer.release()
cv2.destroyAllWindows()
time.sleep(10)
cap = cv2.VideoCapture(outputfile)

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=10)

# Store selected frames in an array
frames = []
for fid in frameIds:

    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
    print(fid)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
# mask= np.zeros_like(LastFrame)
# cv2.fillPoly(mask, pts=[boundZone], color=(255, 255, 255))
medianFrame = cv2.bitwise_and(medianFrame, mask)


cv2.imwrite(outputImage, medianFrame)




medianFrame = cv2.imread(outputImage)
# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)
totalFrame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
videoWidth=cap.get(3)
videoHeight=cap.get(4)
videoFps=cap.get(5)
vid_writer2= cv2.VideoWriter(outputfile2, cv2.VideoWriter_fourcc('M','P','4','V'), videoFps, (int(videoWidth),int(videoHeight)))



sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(medianFrame,None)
# kp2, des2 = orb.detectAndCompute(LastFrame,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
MIN_MATCH_COUNT=4
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
counter=0
while True:
    print("=================")
    
        # Read a new frame
    ok, frame = cap.read()
#     if(counter%10!=0):
#         continue
    if not ok:
         break
        # Start timer
    timer = cv2.getTickCount()


    # if(counter ==0):
    #     kp2, des2 = orb.detectAndCompute(frame,None)
    #     matches = bf.match(des1,des2)
    #     matches = sorted(matches, key = lambda x:x.distance)
    #     p1, p2 = [], []
    #     for f in matches:
    #         p1.append(kp1[f.queryIdx].pt)
    #         p2.append(kp2[f.trainIdx].pt)
    #     H, _ = cv2.findHomography(np.float32(p2), np.float32(p1), cv2.RHO)




    # current_frame = cv2.warpPerspective(frame, H, (frame.shape[1],frame.shape[0]))


    if(counter%30==0 or counter==0):
        kp2, des2 = sift.detectAndCompute(frame,None)
        matches = flann.knnMatch(des1,des2,k=2)
        #matches = sorted(matches, key = lambda x:x.distance)
        p1, p2 = [], []
        sumDist=0
        distCounter=0
        goodMatches = []
        minRatio = 0.7
        for m,n in matches:
            print(m.distance)
            print(n.distance)
            if(m.distance>120 or n.distance>120):
                continue
            if m.distance / n.distance < minRatio:   
                goodMatches.append(m)   
        print(len(goodMatches))
        if len(goodMatches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1,2)
            # H,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
            distSum=0
            for i in range(len(src_pts)):
                point1=(int(src_pts[i][0]),int(src_pts[i][1]))
                point2=(int(dst_pts[i][0]),int(dst_pts[i][1]))
                #distPoint=round(math.dist(point1, point2),2)
                distPoint = round(math.hypot(point2[0] - point1[0], point2[1] - point1[1]), 2)
                distSum+=distPoint
              
            avgDist=distSum/len(src_pts)
            for i in range(len(src_pts)):
                point1=(int(src_pts[i][0]),int(src_pts[i][1]))
                point2=(int(dst_pts[i][0]),int(dst_pts[i][1]))
                distPoint = round(math.hypot(point2[0] - point1[0], point2[1] - point1[1]), 2)
                #distPoint=round(math.dist(point1, point2),2)

                if(src_pts[i][0]>4500 or dst_pts[i][0] >4500):
                    continue
                if(distPoint>avgDist):
                    continue
                p1.append(point1)
                p2.append(point2)
                #frame = cv2.putText(frame, str(distPoint), point1, cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 2, cv2.LINE_AA)

                # frame = cv2.circle(frame, point1, 8, (255,0,0), 2)
                # frame = cv2.circle(frame, point2, 8, (0,0,255), 2)
                # frame = cv2.line(frame, point1,point2, (0,255,0), 6)


            H, _ = cv2.findHomography(np.float32(p2), np.float32(p1), cv2.RHO)

    output_frame = cv2.warpPerspective(frame, H, (frame.shape[1],frame.shape[0]))

    #output_frame = cv2.putText(output_frame, str(avgDist), (250,50), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 3, cv2.LINE_AA)
    # pre_frame=output_frame.copy()


    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    print(str(counter)+"/"+str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    
    vid_writer2.write(output_frame)

    output_frame = cv2.resize(output_frame,None,fx=0.5,fy=0.5)
    cv2.imshow("output", output_frame)
    
    counter+=1
  # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
cap.release()

vid_writer2.release()
cv2.destroyAllWindows()
