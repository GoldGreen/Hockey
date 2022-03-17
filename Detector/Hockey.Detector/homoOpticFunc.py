import cv2
import numpy as np

#pts_src & pts_dst == np.array # Four (or MORE) corners of the 3D court in source image (start top-left corner and go anti-clock wise)
def perspectiveTransformation(img_src,pts_src,img_dst, pts_dst):
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    # Warp source image to destination based on homography
    return cv2.warpPerspective(img_src, h, (img_dst.shape[1], img_dst.shape[0])) #вернёт массив изображения.

#x_src,y_src - from src (3D) img, return [x,y]
def perspectPoingTransormation(x_src,y_src,pts_src, pts_dst):
    # Calculate transfom matrix (h)
	h, status = cv2.findHomography(pts_src, pts_dst)
	# for cv2.perspectiveTransform need this form
	pts = np.array([[[x_src,y_src]]], dtype = "float32")
	arr = cv2.perspectiveTransform(pts,h)
	return np.array(arr).reshape(2)
	 

def getOptivalFlowDirectionV(img, prev_img,step=16):
	prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)[::step,::step] ##Так быстрее, но в должно быть не точно
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[::step,::step]
	
	# Calculates dense optical flow by Farneback method
	flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
									flow=None,
                                    pyr_scale=0.5,
                                    levels=3,
                                    iterations=15,
                                    winsize=3,
                                    poly_n=7,
                                    poly_sigma=1.2,
                                    flags=0) #default 0.5, 3, 15, 3, 5, 1.2, 0)
	
	# Computes the magnitude and angle of the 2D vectors
	# magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
	
	fx, fy = flow[:,:,0], flow[:,:,1] ##step можно сюда вместо верхнего
	magnitude = np.sqrt(fx*fx+fy*fy)
	angle = (np.arctan2(-1*fy,fx ) + np.pi  )*(180/np.pi) 
	#magnitude = cv2.normalize(magnitude, None)
	#вопрос о выборе показательной переменной открыт. пользовался
	#most_freq = np.argmax(np.bincount(angle.flat))
	#arif_mean = np.mean(angle.flat),
	return magnitude, angle #!

	
if __name__ == '__main__':
	path = r"D:\Desktop\Hockey\yolov4-deepsort-master-old\data\video\test_milk2.mp4"
	cap = cv2.VideoCapture(path)


	ret, first_frame = cap.read()
	first_frame =first_frame[:360,480:960]

	#prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
	#prev_gray = prev_gray[::5,::5]
	step = 5
	prev_frame = first_frame
	

	while(cap.isOpened()):
		
		ret, frame = cap.read()
		
		frame=frame[:360,480:960]
		cv2.imshow("input", frame)
		#gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		#gray = gray[::5,::5]
		# Calculates dense optical flow by Farneback method
		magnitude,angle = getOptivalFlowDirectionV(frame,prev_frame,step)
		prev_frame = frame

		
		mask = np.zeros_like(first_frame[::step,::step])
		# Sets image saturation to maximum
		mask[..., 1] = 255
		mask[..., 0] = angle 
		mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

		angleZ = np.array(angle,dtype=int)
		s = np.argmax(np.bincount(angleZ.flat)) #Врёт со средними. В тестовых скриптах всё ОК, но не тут 
		arif = np.mean(angle.flat)
		#vonePNT = mask[180,120,0]
		str = "Most Common angle: {0} | Аverage: {1} | CenterUp point: {2}   \n ".format(s,arif,3)
		print(str);
		
		rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
		cv2.imshow("dense optical flow", rgb)
		
		# Frames are read by intervals of 1 millisecond. The
		# programs breaks out of the while loop when the
		# user presses the 'q' key
		
		if cv2.waitKey() & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
 


		


