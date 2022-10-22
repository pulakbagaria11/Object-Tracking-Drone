from Utils import *
import cv2
import time
import datetime



def recorder():
    c = datetime.datetime.now()
    c = str(c)
    c.replace(':', '-')
    c = c[0:-7]
    inc = 0
    l1 = []
    for l in c:
        l1.append(l)
    c = ""
    for i in l1:
        if i == ':':
            c = c + '-'
        else:
            c = c + i
        inc += 1
    videowriter = cv2.VideoWriter(c + '.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))
    return videowriter
w, h = 360, 240
pid = [0.4, 0.4, 0]
vr = 0
pError = 0
startCounter = 0  # for no Flight 1   - for flight 0
thres = 0.70
myDrone = initializeTello()
classNames= []
classFile = 'coco.names'
print(open(classFile))
with open(classFile,'rt') as f:
    classNames = [line.rstrip() for line in f]
print(classNames[:5])
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:





    img = telloGetFrame(myDrone)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image', img)

    key = cv2.waitKey(1) & 0xFF
    if key==32:
        if startCounter==0:
            startCounter=1
            myDrone.takeoff()
        elif startCounter==1:
            myDrone.land()
            startCounter=0
    elif key==ord('q'):
        myDrone.rotate_counter_clockwise(30)
    elif key==ord('e'):
        myDrone.rotate_clockwise(30)
    elif key==ord('w'):
        myDrone.move_forward(50)
    elif key==ord('a'):
        myDrone.move_left(50)
    elif key==ord('s'):
        myDrone.move_back(50)
    elif key==ord('d'):
        myDrone.move_right(50)
    elif key==56:
        myDrone.move_up(50)
    elif key==50:
        myDrone.move_down(50)
    elif key==16:
        myDrone.emergency()
    elif key==27:
        myDrone.land()
        break
    elif key==ord('c'):
        c = datetime.datetime.now()
        c = str(c)
        c.replace(':', '-')
        c = c[0:-7]
        inc = 0
        l1 = []
        for l in c:
            l1.append(l)
        c = ""
        for i in l1:
            if i == ':':
                c = c + '-'
            else:
                c = c + i
            inc += 1
        cv2.imwrite(c+'.jpg', img)
    elif key==ord('r'):
        if vr==0:
            vr=1
            res = recorder()
            res.write(img)

        else:
            vr=0
            res.release()

