import numpy as np
from geometry import Geometry

# Bubble Sheet Labels
D_UNIT_LABELS_COL1 = np.array(["EX5", "EX8", "EX9", "EX11"])
D_UNIT_LABELS_COL2 = np.array(["SH1", "SH2", "SH3", "SH4", "L"])
OT_LABELS = np.array(["W", "H", "M", "L", "S"])

# Values Indices
D_UNITS_COL1 = np.array([
    [46, 206],
    [46, 220],
    [46, 235],
    [46, 247],
    [46, 262],
])

D_UNITS_COL2 = np.array([
    [175, 206],
    [175, 220],
    [175, 235],
    [175, 247],
    [175, 262],
])

OT_UNITS = np.array([
    [320, 206],
    [320, 220],
    [320, 235],
    [320, 247],
    [320, 262],
])

PA_UNITS = np.array([
    [414, 205],
    [414, 219],
    [414, 234],
    [414, 246],
    [414, 261],
])

DU_UNITS = np.array([
    [455, 205],
    [455, 219],
    [455, 234],
    [455, 246],
    [455, 261],
])

# Bubble Sheet Structure Cordinates
NAME_BLOCK_CORDS = (143, 70, 167, 23)

CREW_BOXES_CORDS = np.array([
    (210, 138, 15, 15),
    (227, 138, 15, 15),
    (245, 138, 15, 15)
])

SHIFT_BOXES_CORDS = np.array([
    (190, 118, 15, 15),
    (247, 118, 15, 15)
])

ID_BOXES_CORDS = Geometry.flatten(np.array([
    Geometry.make_digit_column(15, 46),  # digit 1
    Geometry.make_digit_column(32, 46),  # digit 2
    Geometry.make_digit_column(50, 46),  # digit 3
    Geometry.make_digit_column(70, 46),  # digit 4
]))

NO_BOXES_CORDS = Geometry.flatten(np.array([
    Geometry.make_digit_column(102, 46), # digit 1
    Geometry.make_digit_column(121, 46), # digit 2
]))

VERTICAL_SHIFT = np.array([0, 84, 167, 250, 333, 416])
D_UNITS_COL1_CORDS = Geometry.flatten(Geometry.make_row_paragraph(D_UNITS_COL1, VERTICAL_SHIFT))
D_UNITS_COL2_CORDS = Geometry.flatten(Geometry.make_row_paragraph(D_UNITS_COL2, VERTICAL_SHIFT, count=5))
OT_UNITS_CORDS = Geometry.flatten(Geometry.make_row_paragraph(OT_UNITS, VERTICAL_SHIFT, count=5, gap=18))
PA_UNITS_CORDS = Geometry.flatten(Geometry.make_row_paragraph(PA_UNITS, VERTICAL_SHIFT, count=3, gap=13, ratio=1, w=10))
DU_UNITS_CORDS = Geometry.flatten(Geometry.make_row_paragraph(DU_UNITS, VERTICAL_SHIFT, count=1, gap=0, ratio=1, w=21))
