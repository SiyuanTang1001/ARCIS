#ou zheng
#03/31/2019
import cv2
import sys
import numpy as np
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
    frame_float = frame.astype(np.float32)
    for i in range(number):
        bbox = cv2.selectROI(frameRize, False)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frameRize, p1, p2, (255,0,0), 2, 1)
        cv2.circle(frameRize, (int(bbox[0] + bbox[2]/2),int(bbox[1] + bbox[3]/2)), 5, (0,0,255), 3)
        tmpCenterPoints.append((int(bbox[0]/rate + bbox[2]/2/rate),int(bbox[1]/rate + bbox[3]/2/rate)))
        tmpBbox=(int(bbox[0]/rate),int(bbox[1]/rate),int(bbox[2]/rate),int(bbox[3]/rate))
        tmpTrackers.append(createTrackerByName(Type))

        ok = tmpTrackers[i].init(frame, tmpBbox)
    return tmpTrackers,tmpCenterPoints

def fixVideo(trackers,frame):
        # Draw bounding box
    rows,cols = frame.shape[:2]
    newCenterPoints=[]
    for i in range(len(trackers)):
        ok, bbox = trackers[i].update(frame)

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        #cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        #cv2.circle(frame, (int(bbox[0] + bbox[2]/2),int(bbox[1] + bbox[3]/2)), 5, (0,0,255), 3)
        newCenterPoints.append((int(bbox[0] + bbox[2]/2),int(bbox[1] + bbox[3]/2)))
    pts2 = np.float32(newCenterPoints)
    #M = cv2.getPerspectiveTransform(pts2,pts1)
    M, status = cv2.findHomography(pts2, pts1)
    dst = cv2.warpPerspective(frame,M,(cols,rows))
    return dst

    # Define an initial bounding box
videoName="DJI_20250408122911_0007.MP4"
videoPath = "./input/"+videoName
outputfile="./output_stable_1/"+videoName[:-4]+"stabilization.mp4"
# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)
# Read first frame
print(outputfile)

# Read first frame

success, frame = cap.read()

rows,cols = frame.shape[:2]
inputWidth=cols
inputHeight=rows
videoFPS= cap.get(cv2.CAP_PROP_FPS)
vid_writer = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc('M','J','P','G'), videoFPS, (round(inputWidth),round(inputHeight)))


#bbox = (287, 23, 86, 320)
trackerType = "CSRT"
trackers,initPoints=createLandMarker(5,frame,trackerType,0.5)
pts1 = np.float32(initPoints)
# tracker=createTrackerByName(trackerType)
# # Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)
# # Set video to load

# # Initialize tracker with first frame and bounding box
# ok = tracker.init(frame, bbox)
counter=0
while True:
    counter+=1
        # Read a new frame
    ok, frame = cap.read()
#     if(counter%10!=0):
#         continue
    if not ok:
         break
        # Start timer
    timer = cv2.getTickCount()
        # Update tracker
    frame=fixVideo(trackers,frame)
            # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    print(str(counter)+"/"+str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    #cv2.putText(frame, trackerType + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        # Display FPS on frame
    #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    #cv2.putText(frame, str(counter)+"/"+str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (250,170,50), 2);
    vid_writer.write(frame.astype(np.uint8))
        # Display result
    frame = cv2.resize(frame,None,fx=0.5,fy=0.5)
    cv2.imshow("Tracking", frame)
  # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
