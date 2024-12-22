import ultralytics
from ultralytics import YOLO
import math
import cv2
import cvzone
from sort import *
import numpy as np
cap=cv2.VideoCapture('C:/Users/HP/Downloads/2165-155327596_small.mp4')
model=YOLO('../YOLO-Weight/yolov8n.pt')
#numbers_of_vehicals=0

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", 
    "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", 
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
    "teddy bear", "hair dryer", "toothbrush"
]
mask=cv2.imread('C:/Users/HP/Downloads/mask.png')

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)



totalCount=[]



frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mask=cv2.resize(mask,(frame_width,frame_height))


while True:
    success,img=cap.read()
    
    
    imgRegion=cv2.bitwise_and(img,mask)
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5))
    
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            
            conf=math.ceil((box.conf[0]*100))/100
            line=cv2.line(img,(0,int(frame_height/2)-40),(frame_width,int(frame_height/2)-40),(200,100,255),5)
            #cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)


             
            cls=classNames[int(box.cls[0])]
            if cls=='car' or cls=='bicycle' or cls=='motorbike'or cls=='truck' or cls=='bus' and conf>0.3:
                print(box.cls[0])
               # numbers_of_vehicals=numbers_of_vehicals+1   
               # cvzone.putTextRect(img,f'{conf} {cls}',(max(35,x1),max(35,y1)),scale=0.7,thickness=1)
                #cvzone.cornerRect(img,(x1,y1,w,h),colorR=(203, 192, 255),colorC=(255,0,0),l=20,t=3,rt=5)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))

    resultsTracker=tracker.update(detections)
    for result in resultsTracker:
        
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        w,h=x2-x1,y2-y1
        
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=4,colorR=(255,0,255))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED) #d=>radius

        if 0<cx<frame_width and int(frame_height/2)-40-20 < cy < int(frame_height/2)-40+20:
            if totalCount.count(id)==0:
                totalCount.append(int(id))
                x=len(totalCount)
                line=cv2.line(img,(0,int(frame_height/2)-40),(frame_width,int(frame_height/2)-40),(0,255,0),5)

    cvzone.putTextRect(img,f'Count: {x}',(50,50))
            


    cv2.imshow("img",img)
    #cv2.imshow("imgRegion",imgRegion)
    cv2.waitKey(1)
