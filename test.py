import os
import cv2 as c
import numpy as np
import matplotlib.pyplot as plt
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve


class ObjectDetection:
    URL = r"https://www.dropbox.com/s/xoomeq2ids9551y/opencv_bootcamp_assets_NB13.zip?dl=1"
    asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB13.zip")

    if not os.path.exists(asset_zip_path):
        print(f"Downloading and extracting assets...", end="")
        urlretrieve(URL, asset_zip_path)

        try:
            with ZipFile(asset_zip_path) as z:
                z.extractall(os.path.split(asset_zip_path)[0])
            print("Done")

        except Exception as e:
            print("\nInvalid file.", e)

    classFile = "coco_class_labels.txt"
    with open(classFile, "r") as fp:
        labels = fp.read().split("\n")

    modelFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
    configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
    net = c.dnn.readNetFromTensorflow(modelFile, configFile)

    def __init__(self):
        pass

    def menu(self):
        while True:
            print("\n*** Object Detection Using OpenCV and TensorFlow with Pre-Trained Model ***")
            print("1. Detect Live Objects Through Camera")
            print("2. Detect Objects Through Input Image")
            print("3. Exit")
            try:
                select = int(input("Select Object Detection Method: "))
                if select == 1:
                    self.liveDetection()
                elif select == 2:
                    self.inputImage()
                elif select == 3:
                    print("Exiting program. Goodbye!")
                    break
                else:
                    print("Invalid selection. Please choose 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def liveDetection(self):
        thresh = 0.5
        net = self.net
        s = 0
        if len(sys.argv) > 1:
            s = sys.argv[1]
        src = c.VideoCapture(s)
        window = "Object Detection Using TensorFlow (Live)"
        c.namedWindow(window)
        print("Press 'q' to quit and return to the menu.")

        while True:
            bool, frame = src.read()
            if not bool:
                print("Error in reading frame")
                break
            frame = c.flip(frame, 1)
            height, width = frame.shape[:2]
            blob = c.dnn.blobFromImage(frame, 1.0, size=(300, 300), mean=[0, 0, 0], swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                labelId = int(detections[0, 0, i, 1])
                confidence = float(detections[0, 0, i, 2])
                if confidence > thresh:
                    x1, y1 = int(detections[0, 0, i, 3] * width), int(detections[0, 0, i, 4] * height)
                    x2, y2 = int(detections[0, 0, i, 5] * width), int(detections[0, 0, i, 6] * height)
                    label = self.labels[labelId] if labelId < len(self.labels) else "Unknown"
                    c.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                              c.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    c.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            c.imshow(window, frame)
            c.waitKey(1)
            key = c.getWindowProperty(window, c.WND_PROP_VISIBLE)
            if key < 1 or (c.waitKey(1) & 0xFF == ord('q')):
                break


        src.release()
        c.destroyAllWindows()


    def inputImage(self):
        objSet = set({})
        objList = list([])
        objDict = dict({})
        thresh = 0.5
        net = self.net

        print("\n** Select pre-loaded Sample Image **")
        im_list = os.listdir("images")  # Corrected path
        print(im_list)
        for i in range(len(im_list)):
            print(f"{i + 1}. {im_list[i]}")
        try:
            num = int(input("Select Image: "))
            num = num - 1
            image = im_list[num]
            imPath = os.path.join("images", image) # Use os.path.join for path construction


            im_Arr = c.imread(imPath)

            if im_Arr is None:  # Check if image loading was successful
                raise FileNotFoundError(f"Could not load image: {imPath}")

            height, width = im_Arr.shape[:2]
            blob = c.dnn.blobFromImage(im_Arr, 1.0, size=(300, 300), mean=[0, 0, 0], swapRB=True, crop=False)
            net.setInput(blob)
            object = net.forward()
            imgExtract = dict({})
            for i in range(object.shape[2]):
                imId = int(object[0, 0, i, 1])
                conf = float(object[0, 0, i, 2])
                if conf > thresh:
                    if not imgExtract.get(self.labels[imId]):
                        imgExtract[self.labels[imId]] = []
                    xtop = int(object[0, 0, i, 3] * width)
                    ytop = int(object[0, 0, i, 4] * height)
                    xbottom = int(object[0, 0, i, 5] * width)
                    ybottom = int(object[0, 0, i, 6] * height)
                    imgExtract[self.labels[imId]].append((xtop, ytop, xbottom, ybottom))
                    c.putText(im_Arr, f"{self.labels[imId]}", (xtop, ytop - 5), c.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 255, 0), 1, c.LINE_AA)
                    c.rectangle(im_Arr, (xtop, ytop), (xbottom, ybottom), (255, 255, 255), 2)
                    objSet.add(self.labels[imId])
                    objList.append(self.labels[imId])
            c.imshow("Image Extraction", im_Arr)
            print("Objects Identified in the Image:")
            for x in objSet:
                objDict[x] = objList.count(x)
            print(objDict)
            c.waitKey(0)
            c.destroyAllWindows()
        except Exception as e:
            print(f"Error: {e}")
            print("Unable to read the input image....")


if __name__ == "__main__":
    Obj_det = ObjectDetection()
    Obj_det.menu()