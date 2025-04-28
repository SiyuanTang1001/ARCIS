#ou zheng
#03/31/2019
import cv2
import sys
import numpy as np
import math
def reLocation(frame,frameCut,pFrame):
	windowHeight, windowWidth = frame.shape[:2]
	# Our operations on the frame come here
	currentFrame = cv2.cvtColor(frameCut, cv2.COLOR_BGR2GRAY) 
    #find right transformation
	m=cv2.estimateRigidTransform(currentFrame,pFrame,fullAffine=True)
	if m is None:
		return frame
    #traslation
	dx=m[0,2]
	dy=m[1,2]
    #ratation
	rotationM=np.arctan2(m[1,0],m[0,0])
    #reconstruct
	m = np.zeros((2,3), np.float32)
	m[0,0] = np.cos(rotationM)
	m[0,1] = -np.sin(rotationM)
	m[1,0] = np.sin(rotationM)
	m[1,1] = np.cos(rotationM)
	m[0,2] = dx
	m[1,2] = dy

	frameStabilized = cv2.warpAffine(frame, m, (windowWidth,windowHeight))
	return frameStabilized

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
    return tracker

def createLandMarker(number,frame,Type,rate):
    frameRize = cv2.resize(frame,None,fx=rate,fy=rate)   
    tmpTrackers=[]
    tmpCenterPoints=[]
    tmpBboxGroup=[]
    for i in range(number):
        bbox = cv2.selectROI(frameRize, False)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frameRize, p1, p2, (255,0,0), 2, 1)
        cv2.circle(frameRize, (int(bbox[0] + bbox[2]/2),int(bbox[1] + bbox[3]/2)), 5, (0,0,255), 3)
        tmpCenterPoints.append((int(bbox[0]/rate + bbox[2]/2/rate),int(bbox[1]/rate + bbox[3]/2/rate)))
        tmpBbox=(int(bbox[0]/rate),int(bbox[1]/rate),int(bbox[2]/rate),int(bbox[3]/rate))
        tmpTrackers.append(createTrackerByName(Type))
        tmpBboxGroup.append(tmpBbox)

        ok = tmpTrackers[i].init(frame, tmpBbox)
    return tmpTrackers,tmpCenterPoints,tmpBboxGroup


def recreateLandMarker(boxes,frame,Type,rate):
    frameRize = cv2.resize(frame,None,fx=rate,fy=rate)   
    tmpTrackers=[]
    for i in range(len(boxes)):
        tmpBbox=boxes[i]
        tmpTrackers.append(createTrackerByName(Type))
        ok = tmpTrackers[i].init(frame, tmpBbox)
    return tmpTrackers

def fixVideo(trackers,frame,displayFrmae):
        # Draw bounding box
    rows,cols = frame.shape[:2]
    newCenterPoints=[]
    move=False
    for i in range(len(trackers)):
        ok, bbox = trackers[i].update(frame)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        #cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        #cv2.circle(displayFrmae, (int(bbox[0] + bbox[2]/2),int(bbox[1] + bbox[3]/2)), 5, (0,0,255), 3)
        tmpX=int(bbox[0] + bbox[2]/2)
        tmpY=int(bbox[1] + bbox[3]/2)
        newCenterPoints.append((tmpX,tmpY))
        
        dist = math.hypot(tmpX - pts1[i][0], tmpY - pts1[i][1])
        #cv2.putText(displayFrmae, str(dist), (tmpX,tmpY), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        if dist >5 :
            move = True
            print("moved frame")





    if(move):
        pts2 = np.float32(newCenterPoints)
        #M = cv2.getPerspectiveTransform(pts2,pts1)
        M, status = cv2.findHomography(pts2, pts1)
        #dst = cv2.warpPerspective(frame,M,(cols,rows))
        return True,M
    else:
        return False,0


    # Define an initial bounding boxs
videoName="DJI_20250408122911_0005stabilization.mp4"
videoPath = "./output_stable_2/"+videoName
outputfile="./output_stable_3/"+videoName[:-4]+"Checked.mp4"
# Create a video capture object to read videos

cap = cv2.VideoCapture(videoPath)

initCounter=0
counter=initCounter


cap.set(cv2.CAP_PROP_POS_FRAMES, counter)




# Read first frameW
print(outputfile)
# Read first frame
success, frame = cap.read()
rows,cols = frame.shape[:2]
inputWidth=cols
inputHeight=rows
videoFPS= cap.get(cv2.CAP_PROP_FPS)
vid_writer = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc('M','P','4','V'), videoFPS, (round(inputWidth),round(inputHeight)))

#bbox = (287, 23, 86, 320)
trackerType = "CSRT"
inittrackers,initPoints,boxes=createLandMarker(5,frame,trackerType,0.5)
pts1 = np.float32(initPoints)
# tracker=createTrackerByName(trackerType)
# # Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)
# # Set video to load

# # Initialize tracker with first frame and bounding box
# ok = tracker.init(frame, bbox)
outputImage="./output_stable_2/DJI_20250408122911_0005stabilizationbg.png"
medianFrame = cv2.imread(outputImage)

# perFrame=frame.copy()
fixedCounter=0
while True:
    
        # Read a new frame
    ok, frame = cap.read()
    displayFrmae=frame.copy()
#     if(counter%10!=0):
#         continue
    if not ok:
         break
        # Start timer
    timer = cv2.getTickCount()

        # Update tracker
    tmpTrackers=inittrackers
    # if(counter%1==0 or counter==0):
    moved=False
    moved,Matrix=fixVideo(tmpTrackers,frame,displayFrmae)
    print(moved)
    if(moved and counter>initCounter ):
        
        fixedCounter+=1
        print("re-locate frame")
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
        #matches = sorted(matches, key = lambda x:x.distance)
        kp2, des2 = sift.detectAndCompute(frame,None)
        matches = flann.knnMatch(des1,des2,k=2)
        p1, p2 = [], []
        sumDist=0
        distCounter=0
        goodMatches = []
        minRatio = 0.7
        for m,n in matches:
            if(m.distance>120 or n.distance>120):
                continue
            if m.distance / n.distance < minRatio:   
                goodMatches.append(m)   
        if len(goodMatches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1,2)
            
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
                if(distPoint>avgDist):
                    continue
                p1.append(point1)
                p2.append(point2)

            H, _ = cv2.findHomography(np.float32(p2), np.float32(p1), cv2.RHO)
            #############################################################
            if H is None:
            	print("Homography not found. Skipping transformation.")
            	continue  # or handle appropriately



        output_frame = cv2.warpPerspective(frame, H, (frame.shape[1],frame.shape[0]))
        frame=output_frame.copy()
        displayFrmae=output_frame.copy()
        inittrackers=recreateLandMarker(boxes,frame,trackerType,0.5)
        pts1 = np.float32(initPoints)



    # for i in range(len(inittrackers)):
    #     cv2.circle(displayFrmae, (int(pts1[i][0]),int(pts1[i][1])), 5, (0,255,0), 3)


    #     frame = cv2.warpPerspective(frame,Matrix,(cols,rows))
    #     displayFrmae = cv2.warpPerspective(displayFrmae,Matrix,(cols,rows))

    #     moved,Matrix=fixVideo(tmpTrackers,frame,displayFrmae)
    #     if(moved):
    #         print("not fiexed")
    #     else:
    #         print("fixed")




    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    print(str(counter)+"/"+str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))+"|"+str(fixedCounter))
    #cv2.putText(displayFrmae, str(counter), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    #cv2.putText(frame, trackerType + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        # Display FPS on frame
    #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    #cv2.putText(frame, str(counter)+"/"+str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (250,170,50), 2);
    vid_writer.write(displayFrmae.astype(np.uint8))
        # Display result
    # perFrame=frame.copy()
    frame = cv2.resize(frame,None,fx=0.5,fy=0.5)
    displayFrmae = cv2.resize(displayFrmae,None,fx=0.5,fy=0.5)
    cv2.imshow("Tracking", displayFrmae)
    cv2.imshow("frame", frame)


    
    
    counter+=1
    
  # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
