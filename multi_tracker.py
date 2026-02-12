import cv2 
import os
from cv2 import legacy


Trackers = {
    'csrt' : cv2.legacy.TrackerCSRT_create,
    'mosse' : cv2.legacy.TrackerMOSSE_create,
    'kcf' : cv2.TrackerKCF_create,
    'boosting' : cv2.legacy.TrackerBoosting_create,
    'goturn' : cv2.TrackerGOTURN_create,
    'nano' : cv2.TrackerNano_create,
    'vit' : cv2.TrackerVit_create
}

Tracker_key = 'boosting'
Tracker = Trackers[Tracker_key]()
roi = None

capture = cv2.VideoCapture('/home/simran/goturn_custom/test_video_2.mp4')

while True: 
    frame = capture.read()[1]

    if frame is None:
        break

    #frame = frame.resize(frame(750,550))

    if roi is not None:
        success, box = Tracker.update(frame)

        if success:
            x,y,w,h = [int(c) for c in box]
            cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2) #specifying the colour and thickness and coordinates of the rectangle bbox
        else:
            print('Tracking failed') 

    cv2.imshow('Tracking',frame)
    k = cv2.waitKey(20)

    if k == ord('s'):
        roi = cv2.selectROI('Tracking',frame)

        Tracker.init(frame, roi)
    
    elif k == ord('q'):
        break

capture.Release()
cv2.destroyAllWindows    