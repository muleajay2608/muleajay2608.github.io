import cv2
import numpy as np
from win32api import GetSystemMetrics

def create_blank():
	(xmax, ymax) = (GetSystemMetrics(0), GetSystemMetrics(1))
	img = np.zeros(shape=[ymax, xmax, 3], dtype=np.uint8)
	x1 = int(xmax / 3)
	x2 = int(2 * x1)

	y1 = int(ymax / 3)
	y2 = int(2 * y1)

	img = cv2.line(img,(x1,0),(x1,ymax),(255,0,0),3)
	img = cv2.line(img,(x2,0),(x2,ymax),(255,0,0),3)
	img = cv2.line(img,(0,y1),(xmax,y1),(255,0,0),3)
	img = cv2.line(img,(0,y2),(xmax,y2),(255,0,0),3)

	p1x = int(x1 / 3)
	p2x = p1x + x1
	p3x = p1x + x2

	p1y = int(y1 / 2)
	p2y = p1y + y1
	p3y = p1y + y2

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img,'1',(p1x,p1y), font, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'2',(p2x,p1y), font, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'3',(p3x,p1y), font, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'4',(p1x,p2y), font, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'5',(p2x,p2y), font, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'6',(p3x,p2y), font, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'7',(p1x,p3y), font, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'8',(p2x,p3y), font, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img,'9',(p3x,p3y), font, 4,(255,255,255),2,cv2.LINE_AA)

	return (img, x1, x2, y1, y2, xmax, ymax)