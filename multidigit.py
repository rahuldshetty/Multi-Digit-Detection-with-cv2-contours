import cv2
import numpy as np 
import pandas as pd 

file = "sample.jpg"

orig_image = cv2.imread(file)

def process(image):
	image= cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
	inv = cv2.bitwise_not(thresh)
	struct = np.ones((3,3),np.uint8)
	dilated = cv2.dilate(inv ,struct,iterations=1)	
	edges = cv2.Canny(dilated,30,200)
	return edges,dilated

def manage_contours(image,orig_image):
	results=[]
	contours,hier = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(orig_image,contours,-1,(255,255,255),2)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		small_image = orig_image[y:y+h,x:x+w]
		results.append(small_image)
	return orig_image,contours,results

def save_and_process_individual_images(ilist):
	for c,img in enumerate(ilist):
		edg,dil = process(img)
		cv2.imwrite(str(c)+".jpg",dil)

def get_processed_images(ilist):
	res = []
	for img in ilist:
		edg,dil = process(img)
		res.append((edg,dil))
	return res 

	
edges,dilated = process(orig_image)
new_image,contours,res = manage_contours(edges,orig_image.copy())
save_and_process_individual_images(res)

