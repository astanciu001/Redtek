import cv2
import PIL.ExifTags
import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration

#Imágen sobre la que vamos a trabajar
im_path = "photo.png"

img = cv2.imread(im_path)

#Probar que se ha cargado correctamente
"""
cv2.imshow("imagen", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#Ancho y alto de la imágen cargada
h,w = img.shape[:2]
img_h = h
img_w = w//2
img_size = (img_h, img_w)

#Separa la imagen en dos
img_left = img[0:img_h, 0:img_w]
img_right = img[0:img_h, img_w:w]

#Usar la función de la libreria stereovision para calibrar
calibration = StereoCalibration(input_folder = 'calib_result')
rectified_pair = calibration.rectify((img_left, img_right))

cv2.imshow("rect_left", rectified_pair[0])
cv2.imshow("rect_right", rectified_pair[1])
cv2.waitKey(0)
cv2.destroyAllWindows()


"""Buscar todos estos parámetros"""

#SAD window Size
SWS = 9
#Prefilter size
PFS = 9
#Prefilter cap
PFC = 27
#MinDisparity
MDS = 4
#NumDisparities
NOD = 32
#Texture threshold
TTH = 11
#Uniqueness Ratio
UR = 1
#Speckle Range
SR = 0
#Speckle Window
SPWS = 0

#FUnción para obtener mapa de profundidad
def stereo_depth_map(rectified_pair):
    c,r = rectified_pair[0].shape[:2]
    print(c,r)
    disparity = np.zeros((c,r), np.uint8)
    """
    StereoBM: class for computing stereo correspondence
    using the block matching algorithm
    """
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    """FilterType:{
        PREFILTER_NORMALIZED_RESPONSE: 0
        PREFILTER_XSOBEL: 1
    }"""
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)

    dmLeft = cv2.cvtColor(rectified_pair[0], cv2.COLOR_BGR2GRAY)
    #print(dmLeft)
    dmRight = cv2.cvtColor(rectified_pair[1], cv2.COLOR_BGR2GRAY)
    cv2.imshow("dmLeft", dmLeft)
    cv2.imshow("dmRight", dmRight)

    #print(dmRight)
    disparity = sbm.compute(dmLeft, dmRight, disparity)
    plt.imshow(disparity)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     #disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
    local_max = disparity.max()
    local_min = disparity.min()
    print ("MAX " + str(local_max))
    print ("MIN " + str(local_min))
    disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
    plt.imshow(disparity_visual)
    plt.show()
    local_max = disparity_visual.max()
    local_min = disparity_visual.min()
    print ("MAX " + str(local_max))
    print ("MIN " + str(local_min))
    #cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
    #disparity_visual = np.array(disparity_visual)
    return disparity

disparity = stereo_depth_map(rectified_pair)
"""
cv2.imshow("mapa_disparidad", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

fig = plt.subplots(1,2)
plt.subplots_adjust(left=0.15, bottom=0.5)
plt.subplot(1,2,1)
dmObject = plt.imshow(rectified_pair[0], 'gray')
plt.subplot(1,2,2)
dmObject = plt.imshow(disparity, aspect='equal', cmap='jet')
plt.show()
