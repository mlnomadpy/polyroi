import jax.numpy as jnp
from jax import vmap
import cv2 as cv


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.round_point()

    def distance(self, point):
        return jnp.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)

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
        old_x = self.x
        self.x = self.x * jnp.cos(theta) - self.y * jnp.sin(theta)
        self.y = old_x * jnp.sin(theta) + self.y * jnp.cos(theta)
        self.round_point()

    def to_cylindrical(self):
        r = jnp.sqrt(self.x**2 + self.y**2)
        theta = jnp.arctan(self.y/self.x)
        return (r, theta)

    def from_cylindrical(self, r, theta):
        self.x = r * jnp.cos(theta)
        self.y = r * jnp.sin(theta)
        self.round_point()

    def round_point(self):
        self.x = float(jnp.round(self.x))
        self.y = float(jnp.round(self.y))

    def to_tuple(self):
        return (self.x, self.y)

    def draw_point(self, frame):
        # Convert to integers for OpenCV
        point_int = (int(self.x), int(self.y))
        cv.circle(frame, point_int, radius=1,
                  color=(255, 0, 255), thickness=1)

    def __str__(self):
        return "X: {} Y: {} \n".format(self.x, self.y)

