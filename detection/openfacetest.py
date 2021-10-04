import cv2
import time
import numpy as np

#model = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

detector = cv2.dnn.readNetFromCaffe("opencvssd/deploy.prototxt" , "opencvssd/res10_300x300_ssd_iter_140000.caffemodel")
#cap = cv2.VideoCapture(0)
'''
while True:
    ret, img = cap.read()
'''
start = time.time()
img = cv2.imread("test1.jpg")
width = int(img.shape[1]/4)
height = int(img.shape[0]/4)
image = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
(h,w) = image.shape[:2]
target_size = (300, 300)
resized = cv2.resize(image, target_size)

imageBlob = cv2.dnn.blobFromImage(resized,1.0,(300,300),(104.0,177.0,123,0))

detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence < 0.60:
        continue

    box = detections[0,0,i,3:7]*np.array([w,h,w,h])
    (startX,startY,endX,endY) = box.astype("int")

    text = "{:2f}%".format(confidence*100)
    y = startY-10 if startY -10 > 10 else startY+10
    cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255), 2)

done = time.time()
elapsed = done - start
print(elapsed)
cv2.imshow("output",image)
cv2.waitKey(0)