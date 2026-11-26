import time
from itertools import product

import cv2
import numpy as np

CUBE_RADIUS = 100
CUBE_COLORS = [
  np.array([98, 97, 96]), # white
  np.array([12, 110, 101]), # yellow
  np.array([8, 10, 71]), # red
  np.array([2, 33, 122]), # orange
  np.array([65, 29, 23]), # blue
  np.array([25, 76, 29]), # green
]
COLOR_IDX_TO_NAME = [
  'white',
  'yellow',
  'red',
  'orange',
  'blue',
  'green',
]
CUBE_CORNER_PIECES_COLORS = [ # clockwise
  [1, 5, 2], # y g r
  [1, 3, 5], # y o g
  [1, 4, 3], # y b o
  [1, 2, 4], # y r b
  [0, 2, 5], # w r g
  [0, 5, 3], # w b r
  [0, 3, 4], # w o b
  [0, 4, 2], # w g o
]
CUBE_EDGE_PIECES_COLORS = [
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 2],
  [1, 3],
  [1, 4],
  [1, 5],
  [2, 4],
  [2, 5],
  [3, 4],
  [3, 5],
]
CUBE_EDGE_POSITIONS = [
  [(0, 5), (2, 1)],  # w r  UR
  [(0, 3), (3, 1)],  # w o  UL
  [(0, 1), (4, 1)],  # w b  UB
  [(0, 7), (5, 1)],  # w g  UF
  [(1, 5), (2, 7)],  # y r  DR
  [(1, 3), (3, 7)],  # y o  DL
  [(1, 7), (4, 7)],  # y b  DB
  [(1, 1), (5, 7)],  # y g  DF
  [(2, 5), (4, 3)],  # r b  BR
  [(2, 3), (5, 5)],  # r g  FR
  [(3, 3), (4, 5)],  # o b  BL
  [(3, 5), (5, 3)],  # o g  FL
]

CUBE_CORNER_POSITIONS = [
  [(1, 2), (5, 8), (2, 6)],  # y g r  DFR
  [(1, 0), (3, 8), (5, 6)],  # y o g  DLF
  [(1, 6), (4, 8), (3, 6)],  # y b o  DBL
  [(1, 8), (2, 8), (4, 6)],  # y r b  DRB
  [(0, 8), (2, 0), (5, 2)],  # w r g  URF
  [(0, 6), (5, 0), (3, 2)],  # w g o  ULF
  [(0, 0), (3, 0), (4, 2)],  # w o b  ULB
  [(0, 2), (4, 0), (2, 2)],  # w b r  UBR
]


def rotate(face: list[int], rotations: int) -> list[int]:
  rotations = rotations % 4

  for _ in range(rotations):
    new_face_idxs = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    face = [face[idx] for idx in new_face_idxs]

  return face


def get_all_rotations(face: list[int]) -> list[list[int]]:
  return [rotate(face, i) for i in range(4)]


def perm_parity(p: list[int]) -> int:
  seen, parity = [False] * len(p), 0
  for i in range(len(p)):
    if not seen[i]:
      j, n = i, 0
      while not seen[j]:
        seen[j] = True
        j = p[j]
        n += 1
      parity ^= (n - 1) & 1
  return parity


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


def bgr_to_hsv(colors) -> np.ndarray:
  return cv2.cvtColor(np.array([colors], np.uint8), cv2.COLOR_BGR2HSV)[0]


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


class CubeAssembler:
  def __init__(self):
    self.faces = [None for _ in range(6)]
    self.oriented = []

  def add_face(self, face: list[int]):
    self.faces[face[4]] = face

  def validate(self) -> tuple[bool, str]:
    self.oriented = []

    missing = [COLOR_IDX_TO_NAME[i] for i in range(6) if self.faces[i] is None]
    if missing:
      return False, f'These cube faces have not been recognized: {missing}'

    # i'm lazy
    for faces in product(*(get_all_rotations(face) for face in self.faces)):
      faces = list(faces)

      corners = self._read_corners(faces)
      if corners is None:
        continue
      corners_perm, corners_parity = corners
      if corners_parity != 0:
        continue

      edges = self._read_edges(faces)
      if edges is None:
        continue
      edges_perm, edges_parity = edges
      if edges_parity != 0:
        continue

      if perm_parity(corners_perm) != perm_parity(edges_perm):
        continue

      # every position matched a distinct piece
      self.oriented.append(faces)

    if len(self.oriented) > 0:
      return True, ''

    return False, 'No possible combination of face orientations are valid.'

  def _read_corners(self, faces: list[list[int]]):
    # checking corners
    corners_accounted = [False for _ in range(8)]
    parity = 0
    perm = []

    for positions in CUBE_CORNER_POSITIONS:
      observed = [faces[f][s] for f, s in positions]

      match = self._find_corner(observed, corners_accounted)
      if match is None:
        return None

      idx, twist = match
      corners_accounted[idx] = True
      parity = (parity + twist) % 3
      perm.append(idx)

    return perm, parity

  @staticmethod
  def _find_corner(observed: list[int], accounted: list[bool]):
    for idx, colors in enumerate(CUBE_CORNER_PIECES_COLORS):
      # clockwise order, so any cyclic rotation is a valid twist
      if accounted[idx]:
        continue

      for i in range(3):
        if observed == (colors[i:] + colors[:i]):
          return idx, i

    return None

  def _read_edges(self, faces: list[list[int]]):
    # checking edges
    edges_accounted = [False for _ in range(12)]
    parity = 0
    perm = []

    for positions in CUBE_EDGE_POSITIONS:
      observed = [faces[f][s] for f, s in positions]

      match = self._find_edge(observed, edges_accounted)
      if match is None:
        return None

      idx, flip = match
      edges_accounted[idx] = True
      parity = (parity + flip) % 2
      perm.append(idx)

    return perm, parity

  @staticmethod
  def _find_edge(observed: list[int], accounted: list[bool]):
    for idx, colors in enumerate(CUBE_EDGE_PIECES_COLORS):
      # both flips of an edge are valid
      if accounted[idx]:
        continue

      if set(observed) == set(colors):
        return idx, (0 if observed == colors else 1)

    return None


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


def main():
  CubeScanner().run()


if __name__ == "__main__":
  main()
