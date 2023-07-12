import cv2
import pyttsx3

classNames = []
classFile = "/home/jrs/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/jrs/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/jrs/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(160,160)
net.setInputScale(1.0/ 120)
net.setInputMean((120, 120, 120))
net.setInputSwapRB(True)

engine = pyttsx3.init()
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[15].id)
            rate = engine.getProperty('rate')
            engine.setProperty('rate', 175)
            str1 = str(className)
            engine.say(str1 + "detected")
            engine.runAndWait()
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=1)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+180,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,320)
    cap.set(4,240)
    #cap.set(10,70)
    

while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2)
        #print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
