
def calibrateCamera():
    print('Calibrating camera ...')
    nx = 9
    ny = 6
    chessimgs = glob.glob('./camera_cal/*.jpg')

    # termination criteria for sub-pixel accuracy in finding chessboard corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    for chessimg in chessimgs:
        img = cv2.imread(chessimg)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        img_size = (gray.shape[1], gray.shape[0])

        if ret == True:
            # If we found corners, draw them! 
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # cv2.imshow('img', img)

            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)


    return cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    