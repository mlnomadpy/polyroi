from numpy import pi, array, cos, sin, dot
import cv2 as cv
import numpy as np
from .Point import Point
confirm_box = False
draw_rectangle = False
box_x = 0
box_y = 0
box_w = 0
box_h = 0
k = 0
p = None


class Shape:
    def __init__(self, points):
        self.points = []
        for p in points:
            point = Point(*p)
            self.points.append(point)
        self.centroid()
    def to_image(self, i, frame):
        cv.imwrite('image{}.jpg'.format(i), self.extract_content(frame))
    def to_array(self):
        return np.array([np.array(p.to_tuple()) for p in self.points])
    
    def centroid(self):
        points = array([array(p.to_tuple())
                        for p in self.points]) / len(self.points)
        self.center = Point(*points.sum(axis=0))

    def translate_x(self, x):
        for point in self.points:
            point.translate_x(x)

    def translate_y(self, y):
        for point in self.points:
            point.translate_y(y)

    # translate the fist point of the shape then the whole shape to the point
    # TODO Translate Shape by its center
    def translate_to(self, x, y):
        point = self.points[0].to_tuple()
        self.to_rectangle()
        
        xx = self.max_x
        yy = self.max_y
        x_distance = x - xx
        y_distance = y - yy
        for p in self.points:
            p.translate_x(x_distance)
            p.translate_y(y_distance)
        self.centroid()

    def rotate_around_center(self, theta):
        P = array([array(list(p.to_tuple())) for p in self.points])
        self.centroid()
        C = self.center.to_tuple()
        C = array([array(list(C)) for i in range(len(self.points))])
        R = array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
        P_res = dot(R, (P - C).T) + C.T
        for i, p in zip(range(len(self.points)), self.points):
            p.x = P_res[:, i][0]
            p.y = P_res[:, i][1]
            p.round_point()
        self.centroid()

    def reposition(self, x, y):
        pass

    def update(self, x, y, theta):
        self.rotate_around_center(theta)
        self.translate_x(x)
        self.translate_y(y)

    def to_rectangle(self):
        """Create a bounding box of the current Shape
        Cite: https://stackoverflow.com/a/30902423/6512445
        Returns:
            [[(x1,y1), (x2,y2)]]: [lower left point's cordinates, upper right point's cordinates]
        """        
        # extract all the points
        pts = self.to_array()
        # find the maximum x 
        xs = pts[:,0]
        ys = pts[:,1]
        self.max_x = np.amax(xs)
        # find the minimum x 
        self.min_x = np.amin(xs)
        # find the maximum y
        self.max_y = np.amax(ys)
        # find the minimum y
        self.min_y = np.amin(ys)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        # return frame[min_x:max_x, max_y:min_y]
        return [(self.min_x, self.min_y), (self.max_x, self.max_y)]
    
    def extract_content(self, frame):
        mask = np.zeros(frame.shape, dtype=np.uint8)
        roi_corners = np.array(
            [[p.to_tuple() for p in self.points]], dtype=np.int32)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = frame.shape[2] 
        
        # get rid of 0 as to difer them from the mask 0
        # which may cause the mask to wipe them out
        # when performing the bitwise and
        frame[np.where(frame == 0)] = 1
        # fill the wanted pixels with white in mask
        ignore_mask_color = (255,)*channel_count
        cv.fillPoly(mask, roi_corners, ignore_mask_color)

        # keep the pixels with True
        masked_image = cv.bitwise_and(frame, mask)
        return masked_image

    def draw_shape(self, frame, color =(0, 255, 255), thickness = 1):
        for i in range(len(self.points)-1):
            self.draw_line(self.points[i].to_tuple(
            ), self.points[i+1].to_tuple(), frame, color, thickness)
        # drawing the last line between the first point
        # and the last point of the shape
        self.draw_line(self.points[0].to_tuple(), self.points[len(
            self.points) - 1].to_tuple(), frame, color, thickness)
    
    
    def get_histogram(self, frame):
        content = self.extract_content(frame)

        histogram_b = cv.calcHist([content], [
                                  0], None, [255], [0, 256]).flatten()
        histogram_g = cv.calcHist([content], [
                                  1], None, [255], [0, 256]).flatten()
        histogram_r = cv.calcHist([content], [
                                  2], None, [255], [0, 256]).flatten()
        histogram = np.array([histogram_r, histogram_g, histogram_b], dtype = int)
        self.histogram = histogram
        return histogram

    # Drawing a line between two Point Objects
    def draw_line(self, p1, p2, frame, color, thickness):
        cv.line(frame, p1, p2, color, thickness)
    
    @classmethod
    def get_roi(cls, frame):
        """This methods will pop the user a window where he will be able to
        extract the region of interest (roi) from the frame passed as argument, 
        than return a Shape Object.

        Args:
            frame (Numpy array): [A numpy array containing the image]

        Returns:
            [Shape]: [Shape object containing all the points of the selected shape]
        """        
        global confirm_box, draw_rectangle, k, p
        # global k
        s = None
        
        cv.namedWindow("Tracker")

        tmp = frame.copy()
        cv.setMouseCallback("Tracker", cls.selectTarget, tmp)
        while not confirm_box:
            tmp2 = tmp.copy()
            if draw_rectangle:
                p.draw_point(tmp2)
            if k == 1 and s == None:
                s = Shape([p.to_tuple()])

            # prevent the addition of multiple instances of the same point
            if k > 1 and len(s.points)+1 == k:
                s.points.append(p)

            if k > 1:
                s.draw_shape(tmp2)

            cv.imshow("Tracker", tmp2)
            if cv.waitKey(30) == ord('c'):
                return s

    @classmethod
    # OpenCV callback function to get the area of the user selected object
    def selectTarget(cls, event, x, y, flags, param):
        global box_x, box_y, box_w, box_h, confirm_box, draw_rectangle, k, p
        # number of points created
        if event == cv.EVENT_LBUTTONUP:
            p = Point(x, y)
            p.draw_point(param)
            k += 1
        return

    @classmethod
    def copy(cls, shape):
        points = [p.to_tuple() for p in shape.points]
        s = cls(points)
        return s

    def __str__(self):
        s = 'Printing the shapes points: \n'
        for p in self.points:
            s += str(p)
        return s
