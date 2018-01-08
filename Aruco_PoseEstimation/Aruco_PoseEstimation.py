#importing required libraries
import sys
import os
import cv2
from cv2 import aruco
import numpy as np

#to start real-time feed
cap = cv2.VideoCapture(0)

#importing aruco dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

#calibration parameters
calibrationFile = "calibrationFileName.xml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
camera_matrix = calibrationParams.getNode("cameraMatrix").mat()
dist_coeffs = calibrationParams.getNode("distCoeffs").mat()

r = calibrationParams.getNode("R").mat()
new_camera_matrix = calibrationParams.getNode("newCameraMatrix").mat()


aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250 )
markerLength = 0.25   # Here, our measurement unit is centimetre.
arucoParams = cv2.aruco.DetectorParameters_create()

def cameraPoseFromHomography(H):
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = np.cross(H1, H2)

    norm1 = np.linalg.norm(H1)
    norm2 = np.linalg.norm(H2)
    tnorm = (norm1 + norm2) / 2.0;

    T = H[:, 2] / tnorm
    return np.mat([H1, H2, H3, T])

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    size = frame.shape


    #print size
    # Our operations on the frame come here

    imgRemapped_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # aruco.detectMarkers() requires gray image
   
    avg1 = np.float32(imgRemapped_gray)
    avg2 = np.float32(imgRemapped_gray)


    res = cv2.aruco.detectMarkers(imgRemapped_gray, aruco_dict, parameters=arucoParams) #Detect aruco
    imgWithAruco = imgRemapped_gray  # assign imRemapped_color to imgWithAruco directly
    if len(res[0]) > 0:
	print res[0]

	#Corner detection    	
	#print res[0][0][0][0][0]," ",res[0][0][0][0][1]
        #print res[0][0][0][1][0]," ",res[0][0][0][1][1]
        #print res[0][0][0][2][0]," ",res[0][0][0][2][1]
        #print res[0][0][0][3][0]," ",res[0][0][0][3][1]

	#converting corners to pixel
	x1 = (res[0][0][0][0][0],res[0][0][0][0][1])
        x2 = (res[0][0][0][1][0],res[0][0][0][1][1])
	x3 = (res[0][0][0][2][0],res[0][0][0][2][1])
	x4 = (res[0][0][0][3][0],res[0][0][0][3][1])

	#Drawing detected frame white color
	cv2.line(imgWithAruco,x1, x2, (255,0,0), 2)
	cv2.line(imgWithAruco,x2, x3, (255,0,0), 2)
	cv2.line(imgWithAruco,x3, x4, (255,0,0), 2)
	cv2.line(imgWithAruco,x4, x1, (255,0,0), 2)

	#font type hershey_simpex
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(imgWithAruco,'Corner 1',x1, font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(imgWithAruco,'Corner 2',x2, font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(imgWithAruco,'Corner 3',x3, font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(imgWithAruco,'Corner 4',x4, font, 1,(255,255,255),2,cv2.LINE_AA)
	

	# Camera internals
 	focal_length = size[1]
	center = (size[1]/2, size[0]/2)
	camera_matrix = np.array(
                         	[[focal_length, 0, center[0]],
                         	[0, focal_length, center[1]],
                         	[0, 0, 1]], dtype = "double"
                         	)

    	if res[1] != None: # if aruco marker detected
		im_src = imgWithAruco
		im_dst = imgWithAruco
		
		pts_dst = np.array([[res[0][0][0][0][0],res[0][0][0][0][1]],[res[0][0][0][1][0],res[0][0][0][1][1]],[res[0][0][0][2][0],res[0][0][0][2][1]],[res[0][0][0][3][0],res[0][0][0][3][1]]])
		pts_src = pts_dst
		h, status = cv2.findHomography(pts_src, pts_dst)

		imgWithAruco = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

		rvec, tvec,_ = cv2.aruco.estimatePoseSingleMarkers(res[0], markerLength, camera_matrix, dist_coeffs)

		imgWithAruco = cv2.aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec, tvec, 10)
		cameraPose = cameraPoseFromHomography(h)
		

    cv2.imshow("aruco", imgWithAruco)   # display

    

    if cv2.waitKey(2) & 0xFF == ord('q'):   # if 'q' is pressed, quit.
        	break

