import cv2
import numpy as np 

cam = cv2.VideoCapture(0)

while(cam.isOpened()):

	ret, image = cam.read()
	if(not(ret)):
		break
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	retval, thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	closed = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)

	_, contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	drawing = np.copy(image)

	for contour in contours:

		if len(contour) < 5:
			continue

		area = cv2.contourArea(contour)

		if area >= 100:
			continue
		bounding_box = cv2.boundingRect(contour)
		
		extend = area / (bounding_box[2] * bounding_box[3])
		
		# reject the contours with big extend
		if extend > 0.8:
			continue
		
		# calculate countour center and draw a dot there
		m = cv2.moments(contour)
		if m['m00'] != 0:
			center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
			cv2.circle(drawing, center, 3, (0, 255, 0), -1)
			# drawing[center] = [0, 255, 0]
		
		# fit an ellipse around the contour and draw it into the image
		ellipse = cv2.fitEllipse(contour)
		cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))

	# plt.figure(figsize=(10, 5))
	cv2.imshow("EVE1", drawing)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()