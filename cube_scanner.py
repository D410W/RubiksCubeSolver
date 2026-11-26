import time

import cv2
import numpy as np

from cube_consts import *
from cube_assembler import CubeAssembler
from cube_palette import CubePalette
from cube_detector import CubeDetector

def draw_cube_box(frame: np.ndarray):
  height, width = frame.shape[:2]

  center = (width//2, height//2)
  bottom_right = (center[0] + CUBE_RADIUS, center[1] + CUBE_RADIUS)
  top_left = (center[0] - CUBE_RADIUS, center[1] - CUBE_RADIUS)

  cv2.rectangle(frame, bottom_right, top_left, color=(0, 0, 0), thickness=3)
  cv2.rectangle(frame, bottom_right, top_left, color=(255, 255, 255), thickness=1)


def read_face_colors(frame: np.ndarray, patch: int = 7) -> list[np.ndarray]:
  height, width = frame.shape[:2]
  center = (width//2, height//2)
  step = (CUBE_RADIUS * 2) // 3

  positions = [(center[0] + dx*step, center[1] + dy*step)
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)]

  colors = []
  for x, y in positions:
    roi = frame[y-patch:y+patch+1, x-patch:x+patch+1]
    colors.append(np.median(roi.reshape(-1, 3), axis=0).astype(np.uint8))

  return colors


class CubeScanner:
  def __init__(self):
    self.palette = CubePalette(CUBE_COLORS)
    self.detector = CubeDetector(self.palette)
    self.assembler = CubeAssembler()
    self.cap = self._open_camera()
    self.prev_tick = cv2.getTickCount()

  @staticmethod
  def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
      raise RuntimeError("Cannot open camera")
    return cap

  def run(self):
    while True:
      ok, frame = self.cap.read()
      if not ok:
        break

      self._draw_fps(frame)

      colors_mask = self.detector.colors_mask(frame)
      self._draw_cube_corners(frame)
      draw_cube_box(frame)

      face_colors = self.detector.read_rotated_face_colors(frame)
      color_indexes = None if face_colors is None else self.palette.classify(face_colors)

      to_display = cv2.hconcat([
        frame,
        cv2.cvtColor(colors_mask, cv2.COLOR_GRAY2RGB)
      ])

      cv2.imshow("Cube Solver", to_display)

      key = cv2.waitKey(1)
      if key == ord("q"):
        break

      self._handle_key(key, frame, to_display, color_indexes)

    self.cap.release()
    cv2.destroyAllWindows()

  def _draw_fps(self, frame: np.ndarray):
    curr_tick = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_tick - self.prev_tick)
    self.prev_tick = curr_tick
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

  def _draw_cube_corners(self, frame: np.ndarray):
    cube = self.detector.describe(frame)
    if cube is None:
      return

    for i, point in enumerate(cube['box_points']):
      x = int(point[0])
      y = int(point[1])
      cv2.circle(frame, (x, y), 20, (0, 60 * i, 60 * i), 5)

  def _handle_key(self, key: int, frame: np.ndarray, to_display: np.ndarray, color_indexes):
    if key == ord("s"):
      self._save_frame(to_display)

    if key == 32 and color_indexes: # space
      self._add_scanned_face(color_indexes)

    if key in [ord(f'{i+1}') for i in range(6)]:
      self._calibrate_color(key - ord('1'), frame)

  @staticmethod
  def _save_frame(image: np.ndarray):
    filename = f"frame_{int(time.time() * 1000)}.png"
    cv2.imwrite(filename, image)
    print(f"Saved {filename}")

  def _add_scanned_face(self, color_indexes: list[int]):
    print("cube")
    print([COLOR_IDX_TO_NAME[i] for i in color_indexes])

    self.assembler.add_face(color_indexes)
    ok, err = self.assembler.validate()
    print(f'{ok}, {len(self.assembler.oriented)}: {err}')

    for solved in self.assembler.oriented:
      print('\nTHIS IS A CUBED SOLVED: \n', [[COLOR_IDX_TO_NAME[col] for col in face] for face in solved])

  def _calibrate_color(self, color_idx: int, frame: np.ndarray):
    self.palette.set_color(color_idx, read_face_colors(frame)[4])
    print(f'Reference color \'{COLOR_IDX_TO_NAME[color_idx]}\' = {self.palette.colors[color_idx]}')
