import cv2
import pyttsx3
import RPi.GPIO as GPIO
import time

#thres = 0.45 # Threshold to detect object
# Define GPIO pins for ultrasonic sensor, buzzer, and vibration motor
TRIG_PIN = 23
ECHO_PIN = 24
BUZZER_PIN = 17
VIBRATION_PIN = 27

classNames = []
classFile = "/home/jrs/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/jrs/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/jrs/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
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
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo

def setup():
    # Set up GPIO mode and pins
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(VIBRATION_PIN, GPIO.OUT)

def distance_measurement():
    # Generate ultrasonic pulse
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.0001)
    GPIO.output(TRIG_PIN, GPIO.LOW)

    # Measure the duration of the pulse
    while GPIO.input(ECHO_PIN) == GPIO.LOW:
        pulse_start = time.time()

    while GPIO.input(ECHO_PIN) == GPIO.HIGH:
        pulse_end = time.time()

    # Calculate distance in centimeters
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance
def main():
    try:
        while True:
            distance = distance_measurement()
            print("Distance:", distance, "cm")

            if distance <= 100:  # Change the threshold as per your requirement (in cm)
                # Turn on buzzer and vibration motor
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                GPIO.output(VIBRATION_PIN, GPIO.HIGH)
            else:
                # Turn off buzzer and vibration motor
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                GPIO.output(VIBRATION_PIN, GPIO.LOW)

            time.sleep(0.1)  # Adjust the delay as per your requirement

    except KeyboardInterrupt:
        GPIO.cleanup()


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)
    

while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2)
        #print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
