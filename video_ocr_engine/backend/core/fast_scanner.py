# File: /workspace/video_ocr_engine/backend/core/fast_scanner.py
import cv2
import numpy as np
from typing import List, Tuple

class FastScanner:
    def __init__(self, diff_threshold: int = 15, min_area: int = 500):
        self.diff_threshold = diff_threshold
        self.min_area = min_area

    def scan_for_changes(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        Fast structural difference to detect if subtitle region changed.
        Focuses on bottom 30% of the screen by default.
        """
        h, w = frame1.shape[:2]
        crop_y = int(h * 0.7)
        
        f1_crop = cv2.cvtColor(frame1[crop_y:h, :], cv2.COLOR_BGR2GRAY)
        f2_crop = cv2.cvtColor(frame2[crop_y:h, :], cv2.COLOR_BGR2GRAY)
        
        f1_blur = cv2.GaussianBlur(f1_crop, (5, 5), 0)
        f2_blur = cv2.GaussianBlur(f2_crop, (5, 5), 0)
        
        diff = cv2.absdiff(f1_blur, f2_blur)
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                return True
        return False
