import cv2
import time
from mtcnn.mtcnn import MTCNN
#import dlib

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
    # return our bounding box coordinates
	return (startX, startY, w, h)

# load dlib's HOG + Linear SVM face detector
#print("[INFO] loading HOG + Linear SVM face detector...")
#detector = dlib.get_frontal_face_detector()

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cnndetector = MTCNN()

img = cv2.imread('test1.jpg')
width = int(img.shape[1]/4)
height = int(img.shape[0]/4)

resized = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

start = time.time()

# detect faces in the image
#blue-opencv green-mtcnn red-hog
cnnfaces = cnndetector.detect_faces(resized)
#cvfaces = face_cascade.detectMultiScale(gray, 1.1, 4)
#hogfaces = detector(rgb, 1)


for face in cnnfaces:
    x, y, width, height = face['box']
    cv2.rectangle(resized, (x, y), (x+width, y+height), (0, 255, 0), 2)


'''
for (x, y, w, h) in cvfaces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
'''

'''
hogboxes = [convert_and_trim_bb(resized, r) for r in hogfaces]
for (x, y, w, h) in hogboxes:
	# draw the bounding box on our image
	cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
'''
done = time.time()
elapsed = done - start
print(elapsed)

cv2.imshow('img', resized)
cv2.waitKey()