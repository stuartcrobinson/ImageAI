import os
import statistics as s
import time
from os import listdir

import cv2

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

detectorY = ObjectDetection()
detectorY.setModelTypeAsYOLOv3()
detectorY.setModelPath("/Users/stuartrobinson/repos/computervision/ImageAI/gitignore/yolo.h5")
detectorY.loadModel()

inputDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/backhand"
outDir = os.path.join(execution_path, "gitignore", "backhand_crop")
os.makedirs(outDir, exist_ok=True)

onlyfiles = [f for f in listdir(inputDir) if f.endswith('.png')]
onlyfiles.sort()

prevBallXY = [0, 0]  # tennisBall
currCropCoords = [0, 0, 0, 0]  #
prevNonZeroCropCoords = [0, 0, 0, 0]  #


def getTennisBallCoordinates(detections, currCropCoords):
    balls = list(filter(lambda x: x['name'] == 'sports ball', detections))
    # balls = list(filter(lambda x: x['name'] == 'person', detections))
    if len(balls) == 0:
        return [0, 0]
    themax = max(balls, key=lambda x: x['percentage_probability'])
    ballCoords = themax['box_points']
    x = s.mean([ballCoords[0], ballCoords[2]])
    y = s.mean([ballCoords[1], ballCoords[3]])
    x = currCropCoords[0] + x
    y = currCropCoords[1] + y
    return [x, y]


for file in onlyfiles:
    filePath = os.path.join(inputDir, file)
    print("file:", filePath)
    print("prevBallXY", prevBallXY)
    im = cv2.imread(filePath, cv2.IMREAD_COLOR)  # , cv2.IMREAD_GRAYSCALE)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = cv2.imread(filePath)
    if prevBallXY == [0, 0]:
        currCropCoords = [0, 0, 0, 0]
    else:
        x1 = max(0, prevBallXY[0] - 250)
        y1 = max(0, prevBallXY[1] - 250)
        x2 = min(im.shape[1], x1 + 500)
        y2 = min(im.shape[0], y1 + 500)
        currCropCoords = [x1, y1, x2, y2]
        im = im[y1:y2, x1:x2, :]
    #
    print("currCropCoords", prevBallXY)
    start_timeY = time.time()
    outDirDir = os.path.join(outDir, "Y")
    os.makedirs(outDirDir, exist_ok=True)
    detectionsY = detectorY.detectObjectsFromImage(
        # input_image=filePath,
        input_image=im,
        input_type='array',
        output_image_path=os.path.join(outDirDir, file + ".jpg"),
        minimum_percentage_probability=30)
    print("\ntookY", time.time() - start_timeY)
    prevBallXY = getTennisBallCoordinates(detectionsY, currCropCoords)
    print("ballXY", prevBallXY)
    if prevBallXY == [0, 0]:
        #try again
        im = cv2.imread(filePath, cv2.IMREAD_COLOR)  # , cv2.IMREAD_GRAYSCALE)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im = cv2.imread(filePath)
        if prevBallXY == [0, 0]:
            currCropCoords = [0, 0, 0, 0]
        else:
            x1 = max(0, prevBallXY[0] - 500)
            y1 = max(0, prevBallXY[1] - 500)
            x2 = min(im.shape[1], x1 + 1000)
            y2 = min(im.shape[0], y1 + 1000)
            currCropCoords = [x1, y1, x2, y2]
            im = im[y1:y2, x1:x2, :]
        #
        print("currCropCoords", prevBallXY)
        start_timeY = time.time()
        outDirDir = os.path.join(outDir, "Y")
        os.makedirs(outDirDir, exist_ok=True)
        detectionsY = detectorY.detectObjectsFromImage(
            # input_image=filePath,
            input_image=im,
            input_type='array',
            output_image_path=os.path.join(outDirDir, file + ".jpg"),
            minimum_percentage_probability=30)
        print("\ntookY", time.time() - start_timeY)
        prevBallXY = getTennisBallCoordinates(detectionsY, currCropCoords)
        print("ballXY", prevBallXY)
    if prevBallXY == [0, 0]:
        #try again
        im = cv2.imread(filePath, cv2.IMREAD_COLOR)  # , cv2.IMREAD_GRAYSCALE)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im = cv2.imread(filePath)
        x1 = prevNonZeroCropCoords[0]
        y1 = prevNonZeroCropCoords[1]
        x2 = prevNonZeroCropCoords[2]
        y2 = prevNonZeroCropCoords[3]
        currCropCoords = [x1, y1, x2, y2]
        im = im[y1:y2, x1:x2, :]
        #
        print("currCropCoords", prevBallXY)
        start_timeY = time.time()
        outDirDir = os.path.join(outDir, "Y")
        os.makedirs(outDirDir, exist_ok=True)
        detectionsY = detectorY.detectObjectsFromImage(
            # input_image=filePath,
            input_image=im,
            input_type='array',
            output_image_path=os.path.join(outDirDir, file + ".jpg"),
            minimum_percentage_probability=30)
        print("\ntookY", time.time() - start_timeY)
        prevBallXY = getTennisBallCoordinates(detectionsY, currCropCoords)
        print("ballXY", prevBallXY)
    if currCropCoords != [0,0,0,0]:
        prevNonZeroCropCoords = currCropCoords
    print("------------------------------------------------------------------------")

# for eachObject in detections:
#     print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
#     print("--------------------------------")

#TODO try this with blurry video