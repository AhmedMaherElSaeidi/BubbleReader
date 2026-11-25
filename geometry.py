import numpy as np

class Geometry:
    
    @staticmethod
    def flatten(arr):
        return np.array([b for col in arr for b in col])


    @staticmethod
    def make_digit_column(x, start_y, w=12, h=12, gap=12, ratio=1.01, count=10):
        boxes = []
        y = start_y
        
        for i in range(count):
            boxes.append((x, int(round(y)), w, h))
            y += gap
            gap *= ratio
            
        return np.array(boxes)


    @staticmethod
    def make_digit_row(start_x, y, count, gap, ratio, w, h):
        boxes = []
        x = start_x
        
        for i in range(count):
            boxes.append((int(round(x)), y, w, h))
            x += gap
            gap *= ratio
            
        return np.array(boxes)


    @staticmethod
    def make_row_paragraph(points, vertical_shifts, count=4, gap=22, ratio=1.03, w=12, h=12):
        boxes = [Geometry.make_digit_row(point[0], point[1] + vertical_shift, count, gap, ratio, w, h) for vertical_shift in vertical_shifts for point in points]
        return np.array(boxes)