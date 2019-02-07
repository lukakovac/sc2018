import math

def vektor(a, b):
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
