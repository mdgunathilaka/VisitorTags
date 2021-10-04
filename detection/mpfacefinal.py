import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minConf=0.5):
        self.minConf = minConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minConf)

    def findFaces(self,frame,draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(self.results)
        bboxes=[]

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                bboxes.append([id, bbox,detection.score])

                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,2, (0, 255, 0), 2)

        return frame, bboxes


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()

    while True:
        _, frame = cap.read()
        frame, bboxes = detector.findFaces(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0), 2)
        cv2.imshow("Image", frame)

        print(bboxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
