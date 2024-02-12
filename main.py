import cv2
import numpy as np

def find_rotation_and_translation(img1, img2):
    height, width, channels = img1.shape
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.1)]
    
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * 180.0 / np.pi
    translation = M[:, 2]
    
    print("Угол вращения:", rotation_angle)
    print("Смещение:", translation)
    
    # Отрисовка ключевых точек на изображениях
    global_point = []
    g_c = cv2.KeyPoint(x=width/2 + 0, y=height/2 + 0, size=150)
    g_p = cv2.KeyPoint(x=width/2 + translation[0], y=height/2 + translation[1], size=150)
    global_point.append(g_c)
    global_point.append(g_p)
     
    img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color = (255, 0, 0))
    img1_with_keypoints = cv2.drawKeypoints(img1_with_keypoints, global_point, 
                                            None, color = (0, 255, 0))
    
    img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color = (0, 0, 255))
    img2_with_keypoints = cv2.drawKeypoints(img2_with_keypoints, global_point, 
                                            None, color = (0, 255, 0))

    # Показ изображений с отрисованными ключевыми точками
    cv2.imshow("Image 1 with keypoints", img1_with_keypoints)
    cv2.imshow("Image 2 with keypoints", img2_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rotation_angle, translation

# Пример использования
img1 = cv2.imread('00.jpg')
img2 = cv2.imread('11.jpg')

rotation_angle, translation = find_rotation_and_translation(img1, img2)

#print("Угол вращения:", rotation_angle)
#print("Смещение:", translation)