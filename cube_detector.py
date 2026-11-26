import cv2
import numpy as np

from cube_consts import *

def add_hsv(hsv: np.ndarray, hue: int, sat: int, val: int) -> np.ndarray:
  hsv = np.array(hsv, np.int16)

  hsv[0] = (hsv[0] + hue) % 180
  hsv[1] = hsv[1] + sat
  hsv[2] = hsv[2] + val

  return np.array(np.clip(hsv, 0, 255), np.uint8)


def dilate(image: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
  kernel = np.ones((ksize, ksize), dtype=np.uint8)
  return cv2.dilate(image, kernel, iterations=iterations)


def erode(image: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
  kernel = np.ones((ksize, ksize), dtype=np.uint8)
  return cv2.erode(image, kernel, iterations=iterations)


class CubeDetector:
  MIN_AREA = 500.0

  def __init__(self, palette: CubePalette):
    self.palette = palette

  def colors_mask(self, frame: np.ndarray) -> np.ndarray:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    for i, ref_col in enumerate(self.palette.hsv_reference()):
      hue_range = 8
      sat_range = 40
      val_range = 40
      if i == 0: # white
        hue_range = 80

      lower_hsv = add_hsv(ref_col, -hue_range, -sat_range, -val_range)
      upper_hsv = add_hsv(ref_col, hue_range, sat_range, val_range)

      if lower_hsv[0] > upper_hsv[0]:
        first_cross = upper_hsv.copy()
        first_cross[0] = 179
        second_cross = lower_hsv.copy()
        second_cross[0] = 0

        mask = mask | cv2.inRange(hsv_frame, lower_hsv, first_cross)
        mask = mask | cv2.inRange(hsv_frame, second_cross, upper_hsv)
      else:
        mask = mask | cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    mask = mask & self._search_area_mask(hsv_frame.shape[:2])
    return dilate(erode(dilate(mask, 3), 3, 5), 3, 4)

  @staticmethod
  def _search_area_mask(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    radius = int(CUBE_RADIUS * 1.2)

    search_mask = np.zeros((rows, cols), np.uint8)
    search_mask[
      rows//2 - radius:rows//2 + radius,
      cols//2 - radius:cols//2 + radius,
    ] = 255
    return search_mask

  def describe(self, frame: np.ndarray) -> dict:
    color_mask = self.colors_mask(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
      return None

    valid_contours = [c for c in contours if cv2.contourArea(c) >= self.MIN_AREA]
    if not valid_contours:
      return None

    square_contour = max(valid_contours, key=cv2.contourArea)

    filled = np.zeros_like(color_mask)
    cv2.drawContours(filled, [square_contour], contourIdx=-1, color=255, thickness=-1)

    solid_contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(solid_contours[0])

    (cx, cy), (w, h), angle = rect

    return {
      "center": (float(cx), float(cy)),
      "width": float(w),
      "height": float(h),
      "scale": float(0.5 * (w + h)),
      "angle_deg": float(angle),
      "box_points": cv2.boxPoints(rect),
    }

  def read_rotated_face_colors(self, frame: np.ndarray) -> list[np.ndarray]:
    cube = self.describe(frame)

    if cube is None:
      return None

    box_points = np.array(cube['box_points'])

    start_pos = box_points[0]
    down_vec = (box_points[3] - box_points[0]) / 3.0
    right_vec = (box_points[1] - box_points[0]) / 3.0

    colors = []

    for y in range(-1, 2):
      for x in range(-1, 2):
        x_mul = x * 0.9 + 1.5
        y_mul = y * 0.9 + 1.5

        pos = start_pos + right_vec * x_mul + down_vec * y_mul

        colors.append(frame[int(pos[1]), int(pos[0])])

        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, (x + 1) * 255 // 2, (y + 1) * 255 // 2), 2)

    return colors
