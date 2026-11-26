import cv2
import numpy as np


def bgr_to_hsv(colors) -> np.ndarray:
  return cv2.cvtColor(np.array([colors], np.uint8), cv2.COLOR_BGR2HSV)[0]


class CubePalette:
  WHITE_SAT_MAX = 70  # out of 255

  def __init__(self, colors: list[np.ndarray]):
    self.colors = list(colors)

  def set_color(self, color_idx: int, bgr: np.ndarray):
    self.colors[color_idx] = bgr

  def hsv_reference(self) -> np.ndarray:
    return bgr_to_hsv(self.colors)

  def classify(self, captured_colors: list[tuple[int, int, int]]) -> list[int]:
    hsv_captured = bgr_to_hsv(captured_colors)
    hsv_reference = self.hsv_reference()

    col_idxs: list[int] = []
    for cap_col in hsv_captured:
      best_idx = -1
      best_dist = float("inf")

      for col_idx, ref_col in enumerate(hsv_reference):
        dist = self._distance(cap_col, ref_col)
        if dist < best_dist:
          best_dist = dist
          best_idx = col_idx

      col_idxs.append(best_idx)

    return col_idxs

  @staticmethod
  def _circular_hue_dist(h1: int, h2: int) -> float:
    diff = abs(int(h1) - int(h2)) % 180
    circular_steps = min(diff, 180 - diff)
    return circular_steps / 180.0

  def _distance(self, cap_hsv, ref_hsv) -> float:
    h1, s1, v1 = (int(x) for x in cap_hsv)
    h2, s2, v2 = (int(x) for x in ref_hsv)
    s1n, v1n, s2n, v2n = s1/255, v1/255, s2/255, v2/255

    if s2 < 65:  # white, compare saturation/brightness only
      return abs(s1n - s2n) + 0.5 * abs(v1n - v2n)

    if s1 < self.WHITE_SAT_MAX: # sticker is too gray to trust its hue
      return 1.0 + (self.WHITE_SAT_MAX / 255 - s1n)

    dh = self._circular_hue_dist(h1, h2)
    return 2.5 * dh + 0.35 * abs(s1n - s2n) + 0.12 * abs(v1n - v2n)
