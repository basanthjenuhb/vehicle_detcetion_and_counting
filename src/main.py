import cv2
import numpy as np
import math
import sys

def draw_previous():
	for (x,y,w,h) in old_boxes:
		cv2.circle(frame,(getCentroid(x,y,w,h)),3,(255,0,255),3)

def getCentroid(x,y,w,h):
	return x+int(w/2),y+int(h/2)

def getCentroids(boxes):
	centroid = []
	for (x,y,w,h) in boxes:centroid.append(getCentroid(x,y,w,h))
	return centroid

def distance(point_1,point_2):
	return (math.sqrt( (point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2 ))

def see_distance(old_centroids,new_centroids):
	i = 0
	distances = []
	while i < len(old_centroids) and i < len(new_centroids):
		distances.append(distance(old_centroids[i],new_centroids[i]))
		i+=1
	print(distances)

def get_closest(p1,new_centroids):
	minimum , index = sys.maxsize , 0
	for i in range(len(new_centroids)):
		d = distance(p1,new_centroids[i])
		if d < minimum:
			minimum = d
			index = i
	return index , minimum , new_centroids[index]

# Function to check if any vehicle corsses the line
def check(height,count,old_centroids,new_centroids,threshold):
	# print(old_centroids)
	# print(new_centroids)
	for p1 in old_centroids:
		if p1[1] >= height:
			index , distance , p2 = get_closest(p1,new_centroids)
			# if distance > threshold:continue
			if p2[1] < height:
				count += 1
				print("count: ",count)
			# print(p1,p2,distance,height)
	return count


# Function to draw bounding box around contours
def drawBoundingBox(opening,frame,count,min_length,width,height):
	global old_boxes , new_boxes
	_,contours,hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	old_boxes , new_boxes = new_boxes[:] , []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if w < min_length or h < min_length: continue
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
		cx , cy = getCentroid(x,y,w,h)
		cv2.circle(frame,(cx,cy),3,(255,0,0),3)
		new_boxes.append((x,y,w,h))
	cv2.line(frame,(0,height),(width,height),(0,255,255),3)
	old_centroids , new_centroids = getCentroids(old_boxes) , getCentroids(new_boxes)
	# draw_previous()
	count = check(height,count,old_centroids,new_centroids,40)
	# print("\n",old_boxes)
	# print(new_boxes,"\n","*"*150,"\n")
	return count , frame

# Function to get the background from frames the by averaging
def getBackground(cam,count):
	ret ,frame = cam.read()
	avg = np.float32(frame)
	for i in range(count):
		ret ,frame = cam.read()
		cv2.accumulateWeighted(frame,avg,0.01)
		res = cv2.convertScaleAbs(avg)
	background = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	return background

# Adding the results to the output
def display_final(frame,size,mask,count):
	roi = frame[0:size[0],0:size[1]]
	bg = cv2.bitwise_and(roi,roi,mask=mask)
	frame[0:size[0],0:size[1]] = bg
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame,str(count),(220,80), font, 2,(0,0,0),3)
	return frame

source = '../videos/CarsDrivingUnderBridge.mp4'
cam = cv2.VideoCapture(source) # Defining the Camera

background = getBackground(cam,700) # Getting the background image
# cv2.imshow('background',background)

cam = cv2.VideoCapture('../videos/CarsDrivingUnderBridge.mp4') # Re-Defining the Camera

kernel1 = np.ones((3,3),np.uint8)
kernel2 = np.ones((5,5),np.uint8)
width , height = background.shape[1] , background.shape[0]
new_boxes , old_boxes ,count = [] , [] , 0

# Car icon to display count at the top
car = cv2.imread('car.png',0)
size = car.shape
ret , mask = cv2.threshold(car,220,255,cv2.THRESH_BINARY)

while(1):
	ret ,frame = cam.read()
	if not ret:
		break
	# Preprocessing the image
	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Conversion to gray-scale
	frame_diff = cv2.absdiff(background,frame_gray) # Background subtraction
	ret , thresh = cv2.threshold(frame_diff,35,255,cv2.THRESH_BINARY) # Thresholding the subtracted frame
	erosion = cv2.erode(thresh, kernel1, iterations=1) # Filtering by erosion
	dilation = cv2.dilate(erosion,kernel2 , iterations=2) # Filling holes by dilation
	median = cv2.medianBlur(dilation,5) # Median filtering for smoothing
	
	count , final = drawBoundingBox(median,frame,count,30,width,int(height/2)) # Drawing contours
	# Displaying the frames
	# cv2.imshow('median',median)
	# cv2.imshow('thresh',thresh)
	# cv2.imshow('erosion',erosion)
	# cv2.imshow('dilate',dilation)
	final = display_final(frame,size,mask,count)
	cv2.imshow('final',final)
	# print(boxes)
	k = cv2.waitKey(100) & 0xFF
	if k == ord('q'):break

cv2.destroyAllWindows()
cam.release()
