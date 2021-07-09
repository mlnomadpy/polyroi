from numpy import cos, sin, sqrt, arctan, array
import cv2 as cv


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.round_point()

    def distance(self, point):
        return sqrt((self.x - point.x)**2 + (self.y - point.y)**2)

    def translate_x(self, x):
        self.x += x
        self.round_point()

    def translate_y(self, y):
        self.y += y
        self.round_point()

    def update(self, x, y, theta):
        self.translate_x(x)
        self.translate_y(y)
        self.rotate(theta)

    def rotate(self, theta):
        self.x = self.x * cos(theta) - self.y * sin(theta)
        self.y = self.x * sin(theta) + self.y * cos(theta)
        self.round_point()

    def to_cylindrical(self):
        r = sqrt(self.x**2 + self.y**2)
        theta = arctan(self.y/self.x)
        return (r, theta)

    def from_cylindrical(self, r, theta):
        self.x = r * cos(theta)
        self.y = r * sin(theta)
        self.round_point()

    def round_point(self):
        self.x = round(self.x)
        self.y = round(self.y)

    def to_tuple(self):
        return (self.x, self.y)

    def draw_point(self, frame):
        cv.circle(frame, self.to_tuple(), radius=1,
                  color=(255, 0, 255), thickness=1)

    def __str__(self):
        return "X: {} Y: {} \n".format(self.x, self.y)

