import os
import time

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

detectorY = ObjectDetection()
detectorYT = ObjectDetection()
detectorR = ObjectDetection()

detectorY.setModelTypeAsYOLOv3()
detectorYT.setModelTypeAsTinyYOLOv3()
detectorR.setModelTypeAsRetinaNet()

detectorR.setModelPath("/Users/stuartrobinson/repos/computervision/ImageAI/gitignore/resnet50_coco_best_v2.0.1.h5")
detectorYT.setModelPath("/Users/stuartrobinson/repos/computervision/ImageAI/gitignore/yolo-tiny.h5")
detectorY.setModelPath("/Users/stuartrobinson/repos/computervision/ImageAI/gitignore/yolo.h5")

detectorY.loadModel()
detectorYT.loadModel()
detectorR.loadModel()

# inputImage = os.path.join(execution_path, "images", "image3.jpg")
# inputImage = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/backhand/000001.png"

import cv2

inputDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/forehand"
# inputDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec"
# inputDir = "/Users/stuartrobinson/repos/computervision/ImageAI/gitignore/screengrabs"
outDir = os.path.join(execution_path, "gitignore", "screengrabs_out")
os.makedirs(outDir, exist_ok=True)
for file in os.listdir(inputDir):
    if file.endswith(".png"):
        inputImage = os.path.join(inputDir, file)
        print("file:", inputImage)
        # im = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE)
        # im =cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        #
        start_timeY = time.time()
        outDirDir = os.path.join(outDir, "Y")
        os.makedirs(outDirDir, exist_ok=True)
        detectionsY = detectorY.detectObjectsFromImage(
            input_image=inputImage,
            # input_image=im,
            # input_type='array',
            output_image_path=os.path.join(outDirDir, file + ".jpg"),
            minimum_percentage_probability=30)
        print("\ntookY", time.time() - start_timeY)
        # #
        # start_timeYT = time.time()
        # outDirDir = os.path.join(outDir, "YT")
        # os.makedirs(outDirDir, exist_ok=True)
        # detectionsYT = detectorYT.detectObjectsFromImage(
        #     input_image=inputImage,
        #     output_image_path=os.path.join(outDirDir, file + ".jpg"),
        #     minimum_percentage_probability=30)
        # print("\ntookYT", time.time() - start_timeYT)
        # #
        # start_timeR = time.time()
        # outDirDir = os.path.join(outDir, "R")
        # os.makedirs(outDirDir, exist_ok=True)
        # detectionsR = detectorR.detectObjectsFromImage(
        #     input_image=inputImage,
        #     output_image_path=os.path.join(outDirDir, file + ".jpg"),
        #     minimum_percentage_probability=30)
        # print("\ntookR", time.time() - start_timeR)

# for eachObject in detections:
#     print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
#     print("--------------------------------")
