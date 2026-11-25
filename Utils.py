import cv2
import numpy as np
from datetime import datetime

class Utils:
    
    @staticmethod  
    def is_filled(thresh, box, fill_ratio=0.25):
        x, y, w, h = box
        roi = thresh[y:y+h, x:x+w]
        white_pixels = cv2.countNonZero(roi)
        total_pixels = w * h
        return white_pixels / total_pixels >= fill_ratio


    @staticmethod
    def selected_indices(thresh, points_cords, points_per_row):
        selected_indices = []
        NUM_ROWS = len(points_cords) // points_per_row
        for row_idx in range(NUM_ROWS):
            start = row_idx * points_per_row
            end = start + points_per_row
        
            row_boxes = points_cords[start:end]
        
            selected = -1  # means none selected
            for i, box in enumerate(row_boxes):
                if Utils.is_filled(thresh, box):
                    selected = i
                    break
        
            selected_indices.append(selected)

        return selected_indices
    
    
    @staticmethod
    def selected_column(thresh, Boxes_cords, count):
        values = []
        for col_idx in range(count):
            col_boxes = Boxes_cords[col_idx*10:(col_idx+1)*10]
            for digit, box in enumerate(col_boxes):
                if Utils.is_filled(thresh, box):
                    values.append(str(digit))
                    break
            else:
                values.append("?")
                    
        return "".join(values)

    @staticmethod
    def crop(img, box):
        x,y,w,h = box
        return img[y:y+h, x:x+w]