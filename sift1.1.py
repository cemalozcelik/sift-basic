from matplotlib import pyplot as plt
import cv2

def sift_detect(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:100], None, flags=2)
    (r, g, b) = cv2.split(img3)
    img3= cv2.merge([ b, g, r ])
    return (img3)

def main():
    # load image
    image_a = cv2.imread ('tyt_mat.jpg') # absolute path
    image_b = cv2.imread('table.jpg')
    # SIFT
    img = sift_detect(image_a, image_b)
    plt.imshow(img)
    plt.show()
main()