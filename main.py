import math
import cv2
import numpy as np
from scipy import ndimage
from skimage import color
from skimage.measure import regionprops
from skimage.measure import label
from sklearn.datasets import fetch_mldata
import mathHelper as mathHelper
from operator import  itemgetter


def loadMnist(mnist):
    for i in range(70000):
        img = mnist.data[i].reshape(28, 28)
        grayImg = ((color.rgb2gray(img) / 255.0) > 0.80).astype('uint8')
        slickica = pozicionirajSliku(grayImg)
        mnist_cifre.append(slickica)


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
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 100, 1)
        return min(lines, key=lambda x: x[0][0])[0][0],\
               min(lines, key=lambda x: x[0][0])[0][1],\
               max(lines, key=lambda x: x[0][2])[0][2],\
               max(lines, key=lambda x: x[0][2])[0][3]

def checkIfPointIsOnLine(a, b, c, value):
    #        a Number
    #     /    \
    #   /       \
    # b __Line___c

    distance_ab = mathHelper.vectorMagnitude(mathHelper.vektor(a, b))
    distance_ac = mathHelper.vectorMagnitude(mathHelper.vektor(a, c))
    distance_bc = mathHelper.vectorMagnitude(mathHelper.vektor(b, c))
    # print('AB = ' + str(DistanceAB) + ', AC = ' + str(DistanceAC) + ', BC = ' + str(DistanceBC))
    distance = distance_bc - distance_ab - distance_ac
    # print(str(dist))
    # if value == 4:
    # print(str(dist))
    return distance > -7.5
    # crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    # compare versus epsilon for floating point values, or != 0 if using integers
    # if abs(crossproduct) > 1550:
    #   return False

    # return True

def pozicionirajSliku(img):
    retImg = np.zeros((28, 28), np.uint8)
    try:
        label_img = label(img)
        region = regionprops(label_img)
        x1 = max(region, key=lambda x: x.bbox[2]).bbox[2]
        x2 = min(region, key=lambda x: x.bbox[0]).bbox[0]
        y1 = max(region, key=lambda x: x.bbox[3]).bbox[3]
        y2 = min(region, key=lambda x: x.bbox[1]).bbox[1]

        horz = x1 - x2
        vert = y1 - y2
        retImg[0: horz, 0: vert] = img[x2: x1, y2: y1]
        return retImg
    except ValueError:
        pass


def findNumber(numImg):
    numImg_gray = ((color.rgb2gray(numImg) / 255.0) > 0.90).astype('uint8')
    numImg = pozicionirajSliku(numImg_gray)
    minRazlika = 10000
    rez = -1
    for i in range(len(mnist_cifre)):
        mnistSlika = mnist_cifre[i]
        brRazlika = np.sum(mnistSlika != numImg)
        if brRazlika < minRazlika:
            minRazlika = brRazlika
            rez = mnist.target[i]
    print('REZ = ' + str(rez))
    return rez


def processVideo():
    frejm = 0
    while (1):
        ret, frame = videoCapture.read()
        if not ret:
            break
        maska = cv2.inRange(frame, donja, gornja)

        slikaCB = maska * 1.0
        slikaCB2 = slikaCB

        slikaCB = cv2.dilate(slikaCB, kernel)

        slikaCBLabel, niz = ndimage.label(slikaCB)
        objekti = ndimage.find_objects(slikaCBLabel)

        for i in range(niz):
            centar = []
            duzina = []
            lokacija = objekti[i]

            duzina.append(lokacija[1].stop - lokacija[1].start)
            duzina.append(lokacija[0].stop - lokacija[0].start)

            centar.append((lokacija[1].stop + lokacija[1].start) / 2)
            centar.append((lokacija[0].stop + lokacija[0].start) / 2)

            if duzina[0] > 10 or duzina[1] > 10:
                cifra = {'centar': centar, 'duzina': duzina, 'frejm': frejm}

                rez = []  # cifre koje su blizu
                for cif in cifre:
                    if (tolerancija > mathHelper.vectorMagnitude(mathHelper.vektor(cifra['centar'], cif['centar']))):
                        rez.append(cif)

                if len(rez) == 0:
                    x11 = centar[0] - 14
                    y11 = centar[1] - 14
                    x22 = centar[0] + 14
                    y22 = centar[1] + 14
                    global id
                    id += 1
                    cifra['id'] = id
                    cifra['hasPassed'] = False
                    cifra['value'] = findNumber(slikaCB2[int(y11):int(y22), int(x11):int(x22)])
                    cifra['img'] = slikaCB2[int(y11):int(y22), int(x11):int(x22)]
                    cifre.append(cifra)
                else:
                    najbliziElement = rez[0]
                    min = mathHelper.vectorMagnitude(mathHelper.vektor(najbliziElement['centar'], cifra['centar']))
                    for el in rez:
                        udaljenost = mathHelper.vectorMagnitude(mathHelper.vektor(el['centar'], cifra['centar']))
                        if udaljenost < min:
                            najbliziElement = el
                            min = udaljenost
                    cif = najbliziElement
                    cif['centar'] = cifra['centar']
                    cif['frejm'] = cifra['frejm']

        for cif in cifre:
            # ignore digits that are no more on the screen
            (x, y) = (cif['centar'][0], cif['centar'][1])
            if checkIfPointIsOnLine((x, y), lineEdges[0], lineEdges[1], cif['value']) is True:
                if not cif['hasPassed']:
                    cif['hasPassed'] = True
                    global totalCount
                    totalCount += cif['value']
                    print(format(int(cif['value'])))
        #            if (frejm - cif['frejm'] < 3):
        #                dist, pnt, r = projTackuNaDuz(cif['centar'], ivice[0], ivice[1])
        #               if r > 0:
        #                   if dist < 10:
        #                      if not cif['prosao']:
        #                         cif['prosao'] = True
        #                        global zbir
        #                       zbir += cif['vrednost']
        #                      print (format(int(cif['vrednost'])))
        # cv2.imshow('rgb',slika)
        # cv2.waitKey()
        frejm += 1


outString = "RA 118/2013 Luka Kovac\nfile\tsum"
tolerancija = 15
mnist = fetch_mldata('MNIST original')
mnist_cifre = []
loadMnist(mnist)

for i in range(10):
    totalCount = 0
    videoPath = "video-" + str(i) + ".avi"
    videoCapture = cv2.VideoCapture(videoPath)
    line = detectLineWithHoughP(videoPath)
    if line is not None:
        # find line coordinates
        (x1, y1, x2, y2) = line
        lineEdges = [(x1, y1), (x2, y2)]
        id = -1
        cifre = []
        donja = np.array([160, 160, 160], dtype="uint8")
        gornja = np.array([255, 255, 255], dtype="uint8")
        kernel = np.ones((2, 2), np.uint8)
        processVideo()
        print("totalCount: " + format(int(totalCount)))
        videoCapture.release()

        outString += "\n" + videoPath + "\t" + str(int(totalCount))

f = open('out.txt', 'w')
f.write(outString)
f.close()

print(outString)
