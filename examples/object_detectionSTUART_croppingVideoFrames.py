import json
import os
import statistics as s
import time
from os import listdir

import cv2
from scipy.spatial import distance

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

detectorY = ObjectDetection()
detectorY.setModelTypeAsYOLOv3()
detectorY.setModelPath("/Users/stuartrobinson/repos/computervision/ImageAI/gitignore/yolo.h5")
detectorY.loadModel()


def translateCropCoordToOrig(xy, cropCoords):
    return [xy[0] + cropCoords[0], xy[1] + cropCoords[1]]


def distanceBetween(box_points, ballXY):
    print("in distanceBetween, ", box_points, ", ", ballXY)
    boxX = s.mean([box_points[0], box_points[2]])
    boxY = s.mean([box_points[1], box_points[3]])
    return distance.euclidean((boxX, boxY), (ballXY[0], ballXY[1]))


def translateCropBoxPointsToOrig(XYs, cropCoords):
    return translateCropCoordToOrig([XYs[0], XYs[1]], cropCoords)+ translateCropCoordToOrig([XYs[2], XYs[3]], cropCoords)


def getTennisBallCoordinates(detections, currCropCoords, prevNonZeroBallXY):
    # 'box_points': array([ 47, 125,  68, 166]),
    balls = list(filter(lambda x: x['name'] == 'sports ball', detections))
    if prevNonZeroBallXY != [0, 0]:
        for ball in balls:
            ball['distance'] = distanceBetween(translateCropBoxPointsToOrig(ball['box_points'], currCropCoords), prevNonZeroBallXY)
    pprint.pprint(balls)
    if len(balls) == 0:
        return [0, 0]
    if prevNonZeroBallXY != [0, 0]:
        balls = list(filter(lambda j: j['distance'] < 150, balls))
    if len(balls) == 0:
        return [0, 0]
    themax = max(balls, key=lambda k: k['percentage_probability'])
    ballCoords = themax['box_points']
    x = s.mean([ballCoords[0], ballCoords[2]])
    y = s.mean([ballCoords[1], ballCoords[3]])
    # x = currCropCoords[0] + x
    # y = currCropCoords[1] + y
    return translateCropCoordToOrig([x, y], currCropCoords)


def saveDetections(detections, file):
    with open(file, 'w') as outfile:
        json.dump(json.dump(detections), outfile)


def saveBall(ballXY, file):
    with open(file, 'w') as outfile:
        json.dump(json.loads(str(ballXY)), outfile)


import pprint


def findBall(im, prevBallXY, file, outDir, radius, prevNonZeroCropCoords, prevNonZeroBallXY, name):
    print("radius:", radius)
    currCropCoords = None
    imCrop = None
    if prevNonZeroBallXY == [0, 0]:
        currCropCoords = [0, 0, 0, 0]
        imCrop = im.copy()
    else:
        if radius < 0:
            x1 = prevNonZeroCropCoords[0]
            y1 = prevNonZeroCropCoords[1]
            x2 = prevNonZeroCropCoords[2]
            y2 = prevNonZeroCropCoords[3]
            currCropCoords = [x1, y1, x2, y2]
            imCrop = im[y1:y2, x1:x2, :]
        else:
            x1 = max(0, prevNonZeroBallXY[0] - radius)
            y1 = max(0, prevNonZeroBallXY[1] - radius)
            x2 = min(im.shape[1], x1 + radius * 2)
            y2 = min(im.shape[0], y1 + radius * 2)
            currCropCoords = [x1, y1, x2, y2]
            imCrop = im[y1:y2, x1:x2, :]
    #
    print("currCropCoords", currCropCoords)
    # print("im shape", imCrop.shape)
    print(outDir, file)
    start_timeY = time.time()
    os.makedirs(outDir, exist_ok=True)
    detectionsY = detectorY.detectObjectsFromImage(
        input_image=imCrop,
        input_type='array',
        output_image_path=os.path.join(outDir, file + '_' + str(name) + ".jpg"),
        minimum_percentage_probability=0)
    print("\ntookY", time.time() - start_timeY)
    ballXY = getTennisBallCoordinates(detectionsY, currCropCoords, prevNonZeroBallXY)
    print("ballXY", ballXY)
    print("-----------")
    return detectionsY, currCropCoords, ballXY


# inputDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/backhand"
inputDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec"

outDir = os.path.join(execution_path, "gitignore", "19sec_crop")
os.makedirs(outDir, exist_ok=True)

onlyfiles = [f for f in listdir(inputDir) if f.endswith('.png')]
onlyfiles.sort()

prevBallXY = [0, 0]  # tennisBall
prevNonZeroBallXY = [0, 0]  # tennisBall
currCropCoords = [0, 0, 0, 0]  #
prevNonZeroCropCoords = [0, 0, 0, 0]  #

count = 0
for file in onlyfiles:
    count += 1
    # if count < 100:
    #     continue
    # if 19 < count < 47:
    #     continue
    # # if count < 47:
    # #     continue
    # if count > 113:
    #     break
    print("\n------------------------------------------------------------------------")
    filePath = os.path.join(inputDir, file)
    print("file:", filePath)
    print("prevBallXY", prevBallXY)
    print("prevNonZeroCropCoords", prevNonZeroCropCoords)
    print("prevNonZeroBallXY", prevNonZeroBallXY)
    im = cv2.imread(filePath, cv2.IMREAD_COLOR)  # , cv2.IMREAD_GRAYSCALE)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    detectionsY, currCropCoords, ballXY = findBall(im, prevBallXY, file, outDir, 100, prevNonZeroCropCoords, prevNonZeroBallXY, 1)
    if ballXY == [0, 0]:
        detectionsY, currCropCoords, ballXY = findBall(im, prevBallXY, file, outDir, 150, prevNonZeroCropCoords, prevNonZeroBallXY, 2)
    if ballXY == [0, 0]:
        detectionsY, currCropCoords, ballXY = findBall(im, prevBallXY, file, outDir, 250, prevNonZeroCropCoords, prevNonZeroBallXY, 3)
    # if ballXY == [0, 0]:
    #     detectionsY, currCropCoords, ballXY = findBall(im, prevBallXY, file, outDir, 500, prevNonZeroCropCoords, prevNonZeroBallXY, 3)
    # if ballXY == [0, 0]:
    #     detectionsY, currCropCoords, ballXY = findBall(im, prevBallXY, file, outDir, -1, prevNonZeroCropCoords, prevNonZeroBallXY, 4)
    if currCropCoords != [0, 0, 0, 0]:
        prevNonZeroCropCoords = currCropCoords
    saveBall(ballXY, os.path.join(outDir, file + ".json"))
    prevBallXY = ballXY
    if ballXY != [0, 0]:
        prevNonZeroBallXY = ballXY

print('\nstarting\n')

#TODO 486_3 to 487 has the wrong zoom

#why 477_3 never pick up ball?  or 481?  485 - caught wrong ball