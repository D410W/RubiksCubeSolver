from itertools import product

from cube_consts import *

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
