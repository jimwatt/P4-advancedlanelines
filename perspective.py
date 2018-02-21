import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

saveplots = True

def getPerspectiveTransforms():
	imgrect = mpimg.imread("./test_images/straight_lines1.jpg")
	

	# Hard-code these handselected points from the road
	warpedrectvertices = np.array([
	    [305,650],
	    [525, 500], 
	    [760, 500], 
	    [1000,650]], dtype= np.float32)

	# This is where those hand-selected points defined above ought to be in our bird's-eye view
	offsetx = 160
	offsety = 60
	imx = imgrect.shape[1]
	imy = imgrect.shape[0]
	rectvertices = np.array([
	    [offsetx, imy-offsety],
	    [offsetx, offsety],
	    [imx-offsetx, offsety],
	    [imx-offsetx, imy-offsety]], dtype = np.float32)

	print("warpedrectvertices : ")
	print(warpedrectvertices)

	print("rectvertices : ")
	print(rectvertices)
	

	# Get the perspective transform and its inverse
	M = cv2.getPerspectiveTransform(warpedrectvertices, rectvertices)
	Minv = cv2.getPerspectiveTransform(rectvertices, warpedrectvertices)


	# Plot the perspecetive transform
	if(saveplots):
		warped = cv2.warpPerspective(imgrect, M, (imx, imy))

		cv2.circle(img=imgrect,center=(warpedrectvertices[0,0],warpedrectvertices[0,1]), radius=15, color=(255,0,0), thickness=-1)
		cv2.circle(img=imgrect,center=(warpedrectvertices[1,0],warpedrectvertices[1,1]), radius=15, color=(0,255,0), thickness=-1)
		cv2.circle(img=imgrect,center=(warpedrectvertices[2,0],warpedrectvertices[2,1]), radius=15, color=(0,0,255), thickness=-1)
		cv2.circle(img=imgrect,center=(warpedrectvertices[3,0],warpedrectvertices[3,1]), radius=15, color=(255,255,0), thickness=-1)

		cv2.circle(img=warped,center=(rectvertices[0,0],rectvertices[0,1]), radius=15, color=(255,0,0), thickness=-1)
		cv2.circle(img=warped,center=(rectvertices[1,0],rectvertices[1,1]), radius=15, color=(0,255,0), thickness=-1)
		cv2.circle(img=warped,center=(rectvertices[2,0],rectvertices[2,1]), radius=15, color=(0,0,255), thickness=-1)
		cv2.circle(img=warped,center=(rectvertices[3,0],rectvertices[3,1]), radius=15, color=(255,255,0), thickness=-1)


		plt.figure(1000)
		plt.imshow(imgrect)
		plt.title('Perspective Image')
		plt.savefig('output_images/perspective.png')
		plt.close()

		plt.figure(1001)
		plt.imshow(warped)
		plt.title('Bird\'s-Eye Image')
		plt.savefig('output_images/birdseye.png')
		plt.close()

	return M,Minv