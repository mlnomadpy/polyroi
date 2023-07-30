<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/Y09Ev9i.png" alt="polyroi"></a>
</p>

<h3 align="center">polyroi.py</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![GitHub Issues](https://img.shields.io/github/issues/skywolfmo/polyroi.svg)](https://github.com/skywolfmo/polyroi/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/skywolfmo/polyroi.svg)](https://github.com/skywolfmo/polyroi/pulls)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
  [![Downloads](https://static.pepy.tech/badge/polyroi)](https://pepy.tech/project/polyroi)
  [![Downloads](https://static.pepy.tech/badge/polyroi/month)](https://pepy.tech/project/polyroi)
  [![Downloads](https://static.pepy.tech/badge/polyroi/week)](https://pepy.tech/project/polyroi)
  [![DOI](https://zenodo.org/badge/384454315.svg)](https://zenodo.org/badge/latestdoi/384454315)

</div>

---

<p align="center"> Select and manipulate Region of interest.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

A small python module to select a polygonal region of interest (ROI) in a given image that is stored as a Shape object. You can use this Shape object later to manipulate the polygon selected. You can also extract the inner content from an image, calculate the histogram of the created shape, calculate the center of the shape, rotate the shape around its center, or translate the shape.

## 🏁 Getting Started <a name = "getting_started"></a>

``` python
import cv2 as cv
from polyroi import Shape
```

``` python
img = cv.imread('image.jpg')
shape = Shape.get_roi(img) #returns a Shape object
shape.draw_shape(img, color=(0, 255, 255), thickness=1)
while(1):
    cv.imshow("Getting Started", img)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
```

### Prerequisites

``` shell
pip install cv2
pip install numpy
```

### Installing

``` shell
pip install polyroi 
```


## 🎈 Usage <a name="usage"></a>

Some time ago, I looked for an efficient tool to draw and manipulate polygons in a python environment. But I didn't find anything useful for my case. I did find some tools that can draw and extract a NumPy array, but as for the manipulation of shapes, I had to develop the logic myself. So I decided to create one.
I was trying to implement the particle filter from [Part-Based Lumbar Vertebrae Tracking in Videofluoroscopy Using Particle Filter](https://dblp.org/rec/journals/ijcvip/GuelzimAN20). You can check the repository of how I did manage to work with this package.

``` python
img = cv.imread('image.jpg')

# returns a Shape object
shape = Shape.get_roi(img) 

# Copy the shape
shape2 = Shape.copy(shape) 

# Rotate the shape
shape2.rotate_around_center(np.pi/4) 

# x translate the shape by 5
shape2.translate_x(5) 

# y translate the shape by 5
shape2.translate_y(5) 

# recalculate the center of the shape
shape2.centroid() 

# translate the shape first point to (10, 15) along with the shape
shape2.translate_to(10, 15) 

# x translate, y translate, and rotate around the center by np.pi / 12
shape2.update(5, 3, np.pi / 12) 

# Drawing the shapes
shape.draw_shape(img, color=(0, 255, 255), thickness=1)
shape2.draw_shape(img, color=(0, 255, 0), thickness=1)

# return the bounding box points (upper left, bottom right)
p1, p2 = shape2.to_rectangle() 

# plotting the image
while(1):
    cv.imshow("Getting Started", img)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
```


## ⛏️ Built Using <a name = "built_using"></a>
- [Numpy](https://numpy.org/) - Numpy
- [OpenCV](https://opencv.org/) - OpenCV

## ✍️ Authors <a name = "authors"></a>
- [@skywolfmo](https://github.com/skywolfmo) - Idea & Initial work

See also the list of [contributors](https://github.com/skywolfmo/polyroi/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- https://stackoverflow.com/a/30902423/6512445
- https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region/15343106#15343106
- https://github.com/hbenbel/VOIR
