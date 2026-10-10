from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import cv2, numpy as np

@dataclass
class ColorProfile:
    enabled: bool=False
    rgb: Tuple[int,int,int]=(255,255,255)
    hsv: Tuple[int,int,int]=(0,0,255)
    radius: int=10
    tolerance: float=42.0
    sample_x: float=0.5
    sample_y: float=0.9

    @classmethod
    def from_dict(cls, d: Optional[Dict[str,Any]]) -> "ColorProfile":
        d=d or {}
        rgb=tuple(int(x) for x in (d.get("rgb") or (255,255,255))[:3])
        hsv=tuple(int(x) for x in (d.get("hsv") or (0,0,255))[:3])
        return cls(bool(d.get("enabled",False)),rgb,hsv,int(d.get("radius",10)),float(d.get("tolerance",42)),float(d.get("x",d.get("sample_x",.5))),float(d.get("y",d.get("sample_y",.9))))

class ColorAwarePreprocessor:
    def apply(self, image: np.ndarray, profile: ColorProfile) -> np.ndarray:
        if not profile.enabled or image is None or image.size==0: return image
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        target=np.array(profile.hsv,dtype=np.float32)
        hdist=np.minimum(np.abs(hsv[:,:,0].astype(np.float32)-target[0]),180-np.abs(hsv[:,:,0].astype(np.float32)-target[0]))
        sdist=np.abs(hsv[:,:,1].astype(np.float32)-target[1]); vdist=np.abs(hsv[:,:,2].astype(np.float32)-target[2])
        dist=np.sqrt((hdist*1.1)**2+sdist**2+vdist**2)
        mask=(dist<=profile.tolerance).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced=clahe.apply(gray)
        # Never erase the non-target image completely. The color mask is a guidance channel for
        # the second-pass OCR, not a hard binary segmentation.
        guided=cv2.addWeighted(enhanced,0.55,gray,0.45,0)
        boosted=guided.copy()
        boosted[mask>0]=np.maximum(boosted[mask>0],230)
        return cv2.cvtColor(boosted,cv2.COLOR_GRAY2BGR)
