import cv2
import numpy as np
from PIL import Image

from config import DIM_LIMIT


def resize_img(img):
    max_dim = max(img.shape)
    if max_dim > DIM_LIMIT:
        resize_scale = DIM_LIMIT / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    return img


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def four_point_transform(img, rect):
	(tl, tr, br, bl) = rect
	
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

	return warped


def pil_to_cv2_image(pil_img):
	img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 
	return img_array


def cv2_to_pil_image(img):
	pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	return pil_img
