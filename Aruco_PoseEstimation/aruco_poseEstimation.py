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
	
	#center co-ordinates
        tempx = x1[0]+x2[0]
 	tempx = tempx/2
	tempy = x1[1]+x4[1]
	tempy = tempy/2
	center = (tempx,tempy)
	
        
	#2D image points. If you change the image, you need to change vector
	image_points = np.array([
                            	center,     #center
                            	x1,     # topLeftCorner
                            	x2,     # topRightCorner
                            	x3,     # bottomRightCorner
                            	x4     # bottomLeftCorner
                        	], dtype="double")

	#3D model points. Train your own model
	model_points = np.array([
                            (0.0, 0.0, 0.0),             # center
                            (-225.0, 170.0, -135.0),     # topLeftCorner
                            (225.0, 170.0, -135.0),      # topRightCorner
                            (150.0, -150.0, -125.0),    # bottomRightCorner
                            (-150.0, -150.0, -125.0)      # bottomLeftCorner
                         	
                        ])

	# Camera internals
 	focal_length = size[1]
	center = (size[1]/2, size[0]/2)
	camera_matrix = np.array(
                         	[[focal_length, 0, center[0]],
                         	[0, focal_length, center[1]],
                         	[0, 0, 1]], dtype = "double"
                         	)
       
		
	#print "Camera Matrix :\n {0}".format(camera_matrix)
	#print "Rotation Vector:\n {0}".format(rvec)
	#print "Translation Vector:\n {0}".format(tvec)
	#dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    	if res[1] != None: # if aruco marker detected

		(success, rvec, tvec) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        	rvec, tvec,_ = cv2.aruco.estimatePoseSingleMarkers(res[0], markerLength, camera_matrix, dist_coeffs,rvec, tvec) # posture estimation from a single marker

        	imgWithAruco = cv2.aruco.drawDetectedMarkers(imgRemapped_gray, res[0], res[1], (0,255,0))
        	imgWithAruco = cv2.aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec, tvec, 10)    # axis length 100 can be changed according to your requirement
		
		print "image with aruco ",imgWithAruco.size
    cv2.imshow("aruco", imgWithAruco)   # display

    

    if cv2.waitKey(2) & 0xFF == ord('q'):   # if 'q' is pressed, quit.
        	break

