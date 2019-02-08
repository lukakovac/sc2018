import math

def getVector(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x2 - x1, y2 - y1)


def addVectos(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x2 + x1, y2 + y1)


def multiplyVectors(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x2 * x1, y2 * y1)


def scaleVectorByFactor(a, sc):
    x, y = a
    return (sc * x, sc * y)

def vectorMagnitude(a):
    x, y = a
    return math.sqrt(x * x + y * y)


def unitVector(a):
    x, y = a
    magnitude = vectorMagnitude(a)
    return (x / magnitude, y / magnitude)

def checkIfPointIsOnLine(a, b, c, value):
    #        a Number
    #     /    \
    #   /       \
    # b __Line___c

    vec_ab = getVector(a, b)
    vec_ac = getVector(a, c)
    vec_bc = getVector(b, c)
    distance_ab = vectorMagnitude(vec_ab)
    distance_ac = vectorMagnitude(vec_ac)
    distance_bc = vectorMagnitude(vec_bc)
    # print('AB = ' + str(DistanceAB) + ', AC = ' + str(DistanceAC) + ', BC = ' + str(DistanceBC))
    distance = distance_bc - distance_ab - distance_ac
    # print(str(dist))
    (x1, y1) = vec_bc
    (x2, y2) = vec_ac
    # for points below line
    cross_product = x1 * y2 - y1 * x2

    if( distance > -7.5 and cross_product > -1000):
        return True
    else:
        return False
