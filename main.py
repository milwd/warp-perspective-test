import numpy as np
import cv2


# Imx219-200 distortion matrices
mtx = np.array([[237.21036929,   0.,        479.18796748],
                [0.,         235.54517542, 366.09520559],
                [0.,           0.,           1.]])
dist = np.array([[0.00978868],
                [-0.03383362],
                [0.03214306],
                [-0.00745617]])
# 4 point warp system 
# {
pts1 = np.float32([[515, 345], [855, 345], [0, 530], [1359, 530]])
pts2 = np.float32([[0, 0], [250, 0], [250, 350], [0, 350]])
pts = (pts1, pts2)
# }
# 6 point warp system 
# {
# pts1 = None
# pts2 = np.float32([[], [], [], [], [], []])
# pts = (pts1, pts2)
# }


class WarpTestCase:
    def __init__(self, pts=pts, warpedImageWH=(250, 350), horizental_border=0,width=1640, height=1232, flip=0, disp_width=960, disp_height=720,
                 camera_matrix=mtx, camera_distortion=dist, nPoints=4):
        self.pts = pts
        self.warpedImageWH = warpedImageWH
        self.warpMatrix = cv2.getPerspectiveTransform(pts[0], pts[1])
        self.__width = width
        self.__height = height
        self.__flip = flip
        self.__mtx = camera_matrix
        self.__dist = camera_distortion
        self.__dispW = disp_width
        self.__dispH = disp_height
        self.gpu_mat = cv2.cuda_GpuMat()
        self.image, self.out = None, np.zeros(warpedImageWH)
        self.nPoints = nPoints
        self.i = 0
        self.listt = []
        self.warpMatrix = None
        self.showWarped = False
        self.givePosition = (0, 0)
        try:  # GPU undistortRectifyMap
            self.__mapx = self.__mapy = list(map(cv2.cuda_GpuMat, cv2.fisheye.initUndistortRectifyMap(self.__mtx,
                                                                                              self.__dist,
                                                                                              None,
                                                                                              self.__mtx,
                                                                                              (int(self.__dispH*1.5), int(self.__dispW*1.5)),
                                                                                              5)))
        except:  # CPU undistortRectifyMap
            self.__mapx, self.__mapy = cv2.fisheye.initUndistortRectifyMap(self.__mtx,
                                                                    self.__dist,
                                                                    None,
                                                                    self.__mtx,
                                                                    (int(self.__dispH*1.5), int(self.__dispW*1.5)),
                                                                    5)
        self.camset = f'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(self.__flip)+' ! video/x-raw, width='+str(self.__dispW)+', height='+str(self.__dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
    
    def undistort(self, frame):
        try:  # GPU REMAP
            self.gpu_mat.upload(frame)
            undistorted = cv2.cuda.remap(frame, self.__mapx, self.__mapy, cv2.INTER_LINEAR)
            cpu_undistorted_frame = undistorted.download()
            output = cpu_undistorted_frame[:self.__dispH, :self.__dispW]
        except:  # CPU REMAP
            undistorted = cv2.remap(frame, self.__mapx, self.__mapy, cv2.INTER_LINEAR)
            output = undistorted[:self.__dispH, :self.__dispW]
        return output

    def add_border(self, image, horizontal_border=None):
        horizontalBorderSize = self.horizontal_border if horizontal_border is None else horizontal_border
        bordered = cv2.copyMakeBorder(
            image,
            top=0,
            bottom=0,
            left=horizontalBorderSize,
            right=horizontalBorderSize,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        return bordered

    def warp(self, image):
        try:  # GPU WARP
            self.gpu_mat.upload(image)
            warped_gpu = cv2.cuda.warpPerspective(self.gpu_mat, self.warpMatrix, self.warpedImageWH)
            warped_img = warped_gpu.download()
        except:  # CPU WARP
            warped_img = cv2.warpPerspective(image, self.warpMatrix, self.warpedImageWH)
        return warped_img   

    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.listt.append([x, y])
            self.i += 1
            print("Point "+str(self.i)+" : ", [x, y])
        elif event == cv2.EVENT_MOUSEMOVE:
            self.givePosition = (x, y)
        if self.i == self.nPoints:
            self.showWarped = True
            pts = np.array(self.listt)
            print('selected points : ', pts)
            self.listt = []
            self.warpMatrix, _ = cv2.findHomography(pts, self.pts[1])
            self.i = 0

    def circles(self, img):
        cv2.putText(img, "X: "+str(self.givePosition[0])+" Y: "+str(self.givePosition[1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i in self.listt:
            x, y = i
            cv2.circle(img, (x, y), 4, 255, -1)
        return img

    def calculate_img(self, doUndistort=True, imx=True):
        cap = cv2.VideoCapture(self.camset) if imx else cv2.VideoCapture(0, cv2.CAP_DSHOW)
        n=0
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        while True:
            ret, img = cap.read()
            img = self.undistort(img) if doUndistort else img
            self.image = img
            img = self.circles(img)
            if self.showWarped:
                self.out = self.warp(self.image)
            else:
                self.out = np.zeros((self.warpedImageWH[1], self.warpedImageWH[0], 3))

            cv2.imshow('image', self.image)
            cv2.imshow('out', self.out)
            if cv2.waitKey(1) == ord("s"):
                name = "pic"+str(n)+".jpg"
                n+=1
                cv2.imwrite(name, img)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print('hiii')
    wtc = WarpTestCase()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', wtc.draw)
    wtc.calculate_img(doUndistort=False, imx=False)
