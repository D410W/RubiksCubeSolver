CUBE_RADIUS = 100
CUBE_COLORS = [
  [98, 97, 96], # white
  [12, 110, 101], # yellow
  [8, 10, 71], # red
  [2, 33, 122], # orange
  [65, 29, 23], # blue
  [25, 76, 29], # green
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
