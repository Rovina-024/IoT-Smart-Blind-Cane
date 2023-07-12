import cv2
import time
import pyttsx3

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/jrs/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
    
start_time = time.time()
frame_count = 0

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
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,250,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+180,box[1]+30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,0),2)

    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)


    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2, objects=['bottle','person','refrigerator','laptop','toilet','dining table', 'backpack','chair','cell phone' ,'tv', 'keyboard','umbrella', 'handbag','book', 'bowl','spoon', 'fork', 'knife', 'bed', 'couch', 'mouse','scissors','sink'])
        #print(objectInfo)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Add FPS indicator text to the frame
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
