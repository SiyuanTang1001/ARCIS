from geopy.distance import geodesic as GD
import time
#plot trajectory to validate\n",
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import gmplot
import modin.pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import  distance
import pandas as pd
import csv
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import LineString, Point, LinearRing
from shapely.geometry.polygon import Polygon

from IPython.display import clear_output

import datetime
def getDistanceFromline(point,line):
    p = Point(point)
    route = LineString(line)
    pol_ext = LinearRing(route.coords)
    d = pol_ext.project(p)
    p = pol_ext.interpolate(d)
    closest_point_coords = list(p.coords)[0]
    distance = geodesic(closest_point_coords, point).km*1000
    return distance
def pointInsePoly(point,lons_lats_vect):
    polygon = Polygon(lons_lats_vect) # create polygon
    return point.within(polygon)



#==========================================userInput==================================================================
inputFile='./SW_1000_1500withGPS.csv'
inputLightFile='./SW_1000_1500light.csv'

#stop ;ine
stopPoint1=(27.511821612518922, -80.30729714248406)
stopPoint2=(27.51183945421363, -80.30724886272392)
stopPoint3=(27.51185253812124, -80.30717510197927)
stopLine=[stopPoint1,stopPoint2,stopPoint3]
#--------------------cross to stop ----------------------------------------
#
crossWalkPoint1=(27.511828749197157, -80.30728909585737)
crossWalkPoint2=(27.511696720574975, -80.30724752161947)
crossWalkPoint3=(27.5117442984751, -80.3070731780412)
crossWalkPoint4=(27.511854917013373, -80.30718180750151)
#-----------------------lane1--------------------------
#
lane1Point1=(27.512320432906296, -80.30747335331002)
lane1Point2=(27.512324001229295, -80.30743312017655)
lane1Point3=(27.511713816315286, -80.30719842689811)
lane1Point4=(27.511706679629626, -80.3072507299716)
#-----------------------lane2--------------------------
lane2Point1=(27.51232162234731, -80.30743043796765)
lane2Point2=(27.512343032283287, -80.30736070053635)
lane2Point3=(27.511731658027472, -80.3071421005113)
lane2Point4=(27.511717384657977, -80.30718769806253)
##-----------------------Refer point--------------------------
startTime = datetime.datetime(100,1,1,8,0,0)

#==========================================userInput==================================================================

df = pd.read_csv(inputFile)
dfLight = pd.read_csv(inputLightFile)
max_Frame=  df['frameNumber'].max()
max_Frame_light=  dfLight['frame'].max()

crossWalklats_vect=[crossWalkPoint1[0],crossWalkPoint2[0],crossWalkPoint3[0],crossWalkPoint4[0]]
crossWalklons_vect=[crossWalkPoint1[1],crossWalkPoint2[1],crossWalkPoint3[1],crossWalkPoint4[1]]
crossWalklons_lats_vect = np.column_stack((crossWalklons_vect, crossWalklats_vect)) # Reshape coordinates\

lane1lats_vect=[lane1Point1[0],lane1Point2[0],lane1Point3[0],lane1Point4[0]]
lane1lons_vect=[lane1Point1[1],lane1Point2[1],lane1Point3[1],lane1Point4[1]]
lane1lons_lats_vect = np.column_stack((lane1lons_vect, lane1lats_vect)) # Reshape coordinates\

lane2lats_vect=[lane2Point1[0],lane2Point2[0],lane2Point3[0],lane2Point4[0]]
lane2lons_vect=[lane2Point1[1],lane2Point2[1],lane2Point3[1],lane2Point4[1]]
lane2lons_lats_vect = np.column_stack((lane2lons_vect, lane2lats_vect)) # Reshape coordinates\


frameCounter=2
timeSum=0
while True:
    if(frameCounter>max_Frame):
        break
    tic = time.perf_counter()
    preFrameData=df[df["frameNumber"]==frameCounter-1]
    currentFrameData=df[df["frameNumber"]==frameCounter]
    currentFrameLightData=dfLight[dfLight["frame"]==frameCounter]
    hasPed=False
    pedOnCrosss=False
    currentTime=startTime+ datetime.timedelta(0,frameCounter/30)
    for index,row in currentFrameData.iterrows():
        if(row["class"]==0):
            hasPed=True
            pedGPS=Point(float(row["bLon"]),float(row["bLat"]))
            if(pointInsePoly(pedGPS,crossWalklons_lats_vect)):
                pedOnCrosss=True
    for index,row in currentFrameData.iterrows():
        speed=""
        lane=-1
        currentGPS=(row["bLat"],row["bLon"])
        preInformation=preFrameData[preFrameData["objectID"]==int(row['objectID'])]

        loc=(float(row["bLat"]),float(row["bLon"]))

        locPoint=Point(float(row["bLon"]),float(row["bLat"]))
        


        distanceTostop=getDistanceFromline(loc,stopLine)
        isAfterStopLine=pointInsePoly(locPoint,crossWalklons_lats_vect)
        

        if(isAfterStopLine):
            distanceTostop=distanceTostop*(-1)

        if(pointInsePoly(locPoint,lane1lons_lats_vect)):
            lane=1
        elif(pointInsePoly(locPoint,lane2lons_lats_vect)):
            lane=2
        # if(lane==-1):
        #     continue






        if(preInformation.empty):
            speed=""
            distance=""
        else:
            preGPS=(float(preInformation["bLat"]),float(preInformation["bLon"]))
            locPrePoint=Point(float(preInformation["bLon"]),float(preInformation["bLat"]))
            isPreAfterStopLine=pointInsePoly(locPrePoint,crossWalklons_lats_vect)
            distancePreTostop=getDistanceFromline(preGPS,stopLine)
            if(isPreAfterStopLine):
                distancePreTostop=distancePreTostop*(-1)
            distance=distancePreTostop-distanceTostop
            speed=distance/(1/30)

        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"lane"]=lane
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"isAfterStopLine"]=isAfterStopLine
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"speed(m/s)"]=speed
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"dist2stop"]=distanceTostop

        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"distFromPre(meter)"]=distance
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"hasPed"]=hasPed
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"pedOnCrosss"]=pedOnCrosss
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"upLeftLight"]=currentFrameLightData["upLeftLight"].bool()
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"upRightLight"]=currentFrameLightData["upLeftRight"].bool()
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"bottomLight"]=currentFrameLightData["bottomLight"].bool()
        df.loc[(df["frameNumber"]==frameCounter) & (df["objectID"]==int(row['objectID'])),"timeStamp"]=currentTime

    frameCounter+=1
    toc = time.perf_counter()
    timeDiff=toc - tic
    timeSum+=timeDiff
    fps=timeSum/(frameCounter-2)
    time2finish=round(fps*(max_Frame-frameCounter),2)
    prograss=round((frameCounter/max_Frame)*100,2)
    print(frameCounter,"|",prograss,"%|",time2finish,"s")
outputNme=inputFile[:-4]+"Out.csv"
df.to_csv(outputNme)
