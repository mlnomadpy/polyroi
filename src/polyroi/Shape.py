import jax.numpy as jnp
from jax import vmap
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
        return jnp.array([jnp.array(p.to_tuple()) for p in self.points])
    
    def centroid(self):
        points = jnp.array([jnp.array(p.to_tuple())
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
        P = jnp.array([jnp.array(list(p.to_tuple())) for p in self.points])
        self.centroid()
        C = jnp.array(self.center.to_tuple())
        C = jnp.tile(C, (len(self.points), 1))
        R = jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])
        P_res = jnp.dot(R, (P - C).T) + C.T
        for i, p in zip(range(len(self.points)), self.points):
            p.x = float(P_res[0, i])
            p.y = float(P_res[1, i])
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
        self.max_x = jnp.amax(xs)
        # find the minimum x 
        self.min_x = jnp.amin(xs)
        # find the maximum y
        self.max_y = jnp.amax(ys)
        # find the minimum y
        self.min_y = jnp.amin(ys)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        # return frame[min_x:max_x, max_y:min_y]
        return [(float(self.min_x), float(self.min_y)), (float(self.max_x), float(self.max_y))]
    
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
        # Convert to integers for OpenCV
        p1_int = (int(p1[0]), int(p1[1]))
        p2_int = (int(p2[0]), int(p2[1]))
        cv.line(frame, p1_int, p2_int, color, thickness)
    
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

    # JAX-enabled parallel processing methods
    @staticmethod
    @jnp.vectorize
    def _vectorized_rotation(points_array, theta, center):
        """Vectorized rotation operation for JAX parallel processing"""
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        R = jnp.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        centered_points = points_array - center
        return jnp.dot(R, centered_points.T).T + center

    @classmethod
    def process_multiple_shapes_parallel(cls, shapes, operation, *args):
        """
        Process multiple shapes in parallel using JAX vmap
        
        Args:
            shapes: List of Shape objects
            operation: Operation to perform ('rotate', 'translate', 'scale')
            *args: Arguments for the operation
            
        Returns:
            List of processed Shape objects
        """
        if operation == 'rotate':
            theta = args[0]
            return cls._parallel_rotate_shapes(shapes, theta)
        elif operation == 'translate':
            dx, dy = args[0], args[1]
            return cls._parallel_translate_shapes(shapes, dx, dy)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @classmethod
    def _parallel_rotate_shapes(cls, shapes, theta):
        """Rotate multiple shapes in parallel"""
        processed_shapes = []
        
        for shape in shapes:
            # Create a copy to avoid modifying original
            new_shape = cls.copy(shape)
            new_shape.rotate_around_center(theta)
            processed_shapes.append(new_shape)
            
        return processed_shapes

    @classmethod
    def _parallel_translate_shapes(cls, shapes, dx, dy):
        """Translate multiple shapes in parallel"""
        processed_shapes = []
        
        for shape in shapes:
            # Create a copy to avoid modifying original
            new_shape = cls.copy(shape)
            new_shape.translate_x(dx)
            new_shape.translate_y(dy)
            new_shape.centroid()  # Recalculate centroid after translation
            processed_shapes.append(new_shape)
            
        return processed_shapes

    @classmethod
    def batch_process_with_vmap(cls, shapes_points_list, operation_func):
        """
        Use JAX vmap for true parallel processing of shape operations
        
        Args:
            shapes_points_list: List of point arrays for each shape
            operation_func: JAX-compatible function to apply
            
        Returns:
            Processed point arrays
        """
        # Convert to JAX arrays
        points_arrays = [jnp.array([[p.x, p.y] for p in shape.points]) for shape in shapes_points_list]
        max_points = max(len(arr) for arr in points_arrays)
        
        # Pad arrays to same length for vmap
        padded_arrays = []
        for arr in points_arrays:
            if len(arr) < max_points:
                padding = jnp.zeros((max_points - len(arr), 2))
                arr = jnp.concatenate([arr, padding])
            padded_arrays.append(arr)
        
        batch_array = jnp.stack(padded_arrays)
        
        # Apply operation using vmap
        vectorized_op = vmap(operation_func)
        result = vectorized_op(batch_array)
        
        return result
