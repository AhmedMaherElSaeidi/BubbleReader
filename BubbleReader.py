import cv2
import pandas as pd
from config import *
from Utils import Utils
from datetime import datetime
from DigitReader import DigitReader


class BubbleReader:

    def __init__(self):
        # Labels
        self.D_UNIT_LABELS_COL1 = D_UNIT_LABELS_COL1
        self.D_UNIT_LABELS_COL2 = D_UNIT_LABELS_COL2
        self.OT_LABELS = OT_LABELS

        # Cordinates
        self.NAME_BLOCK_CORDS = NAME_BLOCK_CORDS
        self.CREW_BOXES_CORDS = CREW_BOXES_CORDS
        self.SHIFT_BOXES_CORDS = SHIFT_BOXES_CORDS
        self.ID_BOXES_CORDS = ID_BOXES_CORDS
        self.NO_BOXES_CORDS = NO_BOXES_CORDS
        self.D_UNITS_COL1_CORDS = D_UNITS_COL1_CORDS
        self.D_UNITS_COL2_CORDS = D_UNITS_COL2_CORDS
        self.OT_UNITS_CORDS = OT_UNITS_CORDS
        self.PA_UNITS_CORDS = PA_UNITS_CORDS
        self.DU_UNITS_CORDS = DU_UNITS_CORDS

    def read_image(self, path):
        self.img = cv2.imread(path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Threshold: black pen becomes white, rest is black
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (3, 3), 0)
        _, self.thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return self.img, self.gray, self.thresh

    def save_name(self, save_path='./public'):
        x, y, w, h = self.NAME_BLOCK_CORDS
        name_block = self.img[y:y + h, x:x + w]

        rng = np.random.default_rng()
        random_int = rng.integers(1e9, 1e12)
        img_name = f"{datetime.now():%Y%m%d_%H%M%S}_{random_int}.png"

        cv2.imwrite(f'{save_path}/{img_name}', name_block)
        return img_name

    def read_bubble(self):
        # Reading selected shift
        selected_shift = None
        for i, box in enumerate(self.SHIFT_BOXES_CORDS):
            if Utils.is_filled(self.thresh, box):
                selected_shift = "Day" if i == 0 else "Night"
                break

        # Reading selected crew
        selected_crew = None
        for i, box in enumerate(self.CREW_BOXES_CORDS):
            if Utils.is_filled(self.thresh, box):
                selected_crew = chr(ord('A') + i)  # A, B, or C
                break

        # Reading selected values per row
        D_UNITS_COL1 = [self.D_UNIT_LABELS_COL1[idx] if idx >= 0 else None for idx in
                        Utils.selected_indices(self.thresh, self.D_UNITS_COL1_CORDS, 4)]
        D_UNITS_COL2 = [self.D_UNIT_LABELS_COL2[idx] if idx >= 0 else None for idx in
                        Utils.selected_indices(self.thresh, self.D_UNITS_COL2_CORDS, 5)]
        D_UNITS = [
            D_UNITS_COL1[idx] if D_UNITS_COL1[idx] else D_UNITS_COL2[idx]
            for idx in range(len(D_UNITS_COL1))
        ]
        OT_UNITS = [self.OT_LABELS[idx] if idx >= 0 else None for idx in
                    Utils.selected_indices(self.thresh, self.OT_UNITS_CORDS, 5)]

        # Reading handwritten digits per row
        digit_reader = DigitReader()
        pa_units = digit_reader.read_digits(self.img, self.PA_UNITS_CORDS, 3)
        du_units = digit_reader.read_digits(self.img, self.DU_UNITS_CORDS, 1)

        # Returning result set
        return {
            "ID": Utils.selected_column(self.thresh, self.ID_BOXES_CORDS, 4),
            "Number": Utils.selected_column(self.thresh, self.NO_BOXES_CORDS, 2),
            "Shift": selected_shift,
            "Crew": selected_crew,
            'd_units': D_UNITS,
            'ot_units': OT_UNITS,
            'pa_units': pa_units,
            'du_units': du_units
        }

    def read_bubblesheet(self, bubble_path, save_path='./public'):
        _ = self.read_image(bubble_path)
        img_name = self.save_name()

        body = self.read_bubble()
        body['username'] = img_name

        df = pd.DataFrame(body)
        df = df[df['ot_units'].notna() & df['d_units'].notna()]

        rng = np.random.default_rng()
        random_int = rng.integers(1e9, 1e12)
        file_path = f'{save_path}/results_{datetime.now():%Y%m%d_%H%M%S}_{random_int}.csv'
        df.to_csv(file_path, index=False)

        return file_path
