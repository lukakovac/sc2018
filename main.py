import math
import cv2
import numpy as np
from scipy import ndimage
from skimage import color
from skimage.measure import regionprops
from skimage.measure import label
from sklearn.datasets import fetch_mldata
import mathHelper as mathHelper
import blobHelper as blobHelper

digit_img_size = 28
digit_range_treshold = 15
outString = "RA 118/2013 Luka Kovac1\nfile\tsum"

def getDigitCoordinates(blob_position):
    x11 = blob_position[0] - digit_img_size / 2
    y11 = blob_position[1] - digit_img_size / 2
    x22 = blob_position[0] + digit_img_size / 2
    y22 = blob_position[1] + digit_img_size / 2
    return (x11, y11, x22, y22)

def filterDigitsOnFrame(seq, value):
    for el in seq:
        if value - el['frame_indx'] < 5 and el['hasPassed'] is False: yield el

def loadMnist(mnist):
    for i in range(70000):
        img = mnist.data[i].reshape(digit_img_size, digit_img_size)
        grayImg = ((color.rgb2gray(img) / 255.0) > 0.80).astype('uint8')
        imgPos = getImgPosition(grayImg)
        mnist_cifre.append(imgPos)


def detectLineWithHoughP(videoPath):
    capture = cv2.VideoCapture(videoPath)
    kernel = np.ones((2, 2), np.uint8)

    while (capture.isOpened()):
        retval, frame = capture.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_dilate = cv2.dilate(frame_gray, kernel)
        capture.release()
        # Find edges
        edges = cv2.Canny(frame_dilate, 50, 150, 3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, 100, 1)
        return min(lines, key=lambda x: x[0][0])[0][0],\
               min(lines, key=lambda x: x[0][0])[0][1],\
               max(lines, key=lambda x: x[0][2])[0][2],\
               max(lines, key=lambda x: x[0][2])[0][3]

def getImgPosition(img):
    retImg = np.zeros((28, 28), np.uint8)
    try:
        label_img = label(img)
        region = regionprops(label_img)
        x1 = max(region, key=lambda x: x.bbox[2]).bbox[2]
        x2 = min(region, key=lambda x: x.bbox[0]).bbox[0]
        y1 = max(region, key=lambda x: x.bbox[3]).bbox[3]
        y2 = min(region, key=lambda x: x.bbox[1]).bbox[1]

        retImg[0: x1 - x2, 0: y1 - y2] = img[x2: x1, y2: y1]
        return retImg
    except ValueError:
        pass


def findNumber(numImg):
    numImg_gray = ((color.rgb2gray(numImg) / 255.0) > 0.90).astype('uint8')
    numImg = getImgPosition(numImg_gray)
    minDiff = 10000
    num = -1
    for i in range(len(mnist_cifre)):
        mnistSlika = mnist_cifre[i]
        diff = np.sum(mnistSlika != numImg)
        if diff < minDiff:
            minDiff = diff
            num = mnist.target[i]
    return num


def processVideo():
    frame_indx = 0
    ret, frame = videoCapture.read()
    while ret:
        if not ret:
            break
        lowerBoundry = np.array([160, 160, 160], dtype="uint8")
        upperBoundry = np.array([255, 255, 255], dtype="uint8")
        mask_extracted_digits = cv2.inRange(frame, lowerBoundry, upperBoundry)
        mask_extracted_digits = cv2.dilate(mask_extracted_digits, kernel)
        img_lbl, number_of_objects = ndimage.label(mask_extracted_digits)
        blobs = ndimage.find_objects(img_lbl)
        for i in range(number_of_objects):
            blob_size = blobHelper.getBlobSize(blobs[i])
            blob_position = blobHelper.getBlobPosition(blobs[i])
            if blob_size[0] + blob_size[1] > 20:
                current_digit = {'center': blob_position, 'length': blob_size, 'frame_indx': frame_indx}

                close_digit_hints = []
                for processed_digit in processed_digits:
                    if (digit_range_treshold > mathHelper.vectorMagnitude(mathHelper.getVector(current_digit['center'], processed_digit['center']))):
                        close_digit_hints.append(processed_digit)

                if len(close_digit_hints) == 0:
                    (x11, y11, x22, y22) = getDigitCoordinates(blob_position)
                    global id
                    id += 1
                    current_digit['id'] = id
                    current_digit['hasPassed'] = False
                    current_digit['value'] = findNumber(mask_extracted_digits[int(y11):int(y22), int(x11):int(x22)])
                    current_digit['img'] = mask_extracted_digits[int(y11):int(y22), int(x11):int(x22)]
                    processed_digits.append(current_digit)
                else:
                    processed_digit = min(close_digit_hints, key=lambda x: mathHelper.vectorMagnitude(mathHelper.getVector(x['center'], current_digit['center'])))
                    processed_digit['center'] = current_digit['center']
                    processed_digit['frame_indx'] = current_digit['frame_indx']

        for active_digit in filterDigitsOnFrame(processed_digits, frame_indx):
            # ignore digits that are no more on the screen
            if mathHelper.checkIfPointIsOnLine((active_digit['center'][0], active_digit['center'][1]), lineEdges[0], lineEdges[1], active_digit['value']):
                #print('SHOULD ADD DIGIT ' + str(active_digit['value']) + ', HAS PASSED = ' + str(active_digit['hasPassed']))
                if not active_digit['hasPassed']:
                    active_digit['hasPassed'] = True
                    global totalCount
                    totalCount += active_digit['value']
                    print(format(int(active_digit['value'])))
        #cv2.imshow('rgb',frame)
        #cv2.waitKey()
        frame_indx += 1
        ret, frame = videoCapture.read()

for i in range(10):
    mnist = fetch_mldata('MNIST original')
    mnist_cifre = []
    loadMnist(mnist)
    totalCount = 0
    videoPath = "video-" + str(i) + ".avi"
    videoCapture = cv2.VideoCapture(videoPath)
    line = detectLineWithHoughP(videoPath)
    if line is not None:
        # find line coordinates
        (x1, y1, x2, y2) = line
        lineEdges = [(x1, y1), (x2, y2)]
        id = -1
        processed_digits = []
        kernel = np.ones((2, 2), np.uint8)
        processVideo()
        print("totalCount: " + format(int(totalCount)))
        videoCapture.release()

        # Wasn't sure for test data results. If needed add logic for
        # measuring correctness of output results
        outString += "\n" + videoPath + "\t" + str(int(totalCount))

f = open('out.txt', 'w')
f.write(outString)
f.close()

print(outString)
