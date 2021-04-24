
import cv2
import numpy as np


width, height = 250, 100
orwidth, orheight = 1280, 720


out = np.zeros((width, height))
pts = np.float32([])
# pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0.8*width, height], [0.2*width, height], [0, height]])
listt = []
i = 0


def draw(event, x, y, flags, param):
	global i
	global numOfPoints
	global listt
	global pts
	global out
	global width
	global height
	if event == cv2.EVENT_LBUTTONDBLCLK:
		listt.append([x, y])
		i += 1
		cv2.circle(image, (x, y), 4, 255, -1)
		print([x, y])

	if i == 6:
		pts = np.array(listt)
		print('pts : ', pts)
		listt = []
		matrix, status = cv2.findHomography(pts, pts2)
		out = cv2.warpPerspective(image, matrix, (width, height))
		pts = np.delete(pts, [0, 1, 2, 3, 4, 5])
		i = 0


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

original = cv2.imread('curve--0-7')
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
image = np.copy(original)
while True:
	# cv2.imshow('orig', original)
	cv2.imshow('image', image)
	cv2.imshow('out', out)
	# cv2.imshow('out', out)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break




