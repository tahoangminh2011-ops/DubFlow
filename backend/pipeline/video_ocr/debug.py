from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Any
import cv2


class DebugVisualizer:
    """Optional frame-level debug overlay; never required by the OCR hot path."""
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame, *, frame_index: int, timestamp: float,
              detections: Iterable[Mapping[str, Any]] = (), tracks: Iterable[Mapping[str, Any]] = ()) -> Path:
        canvas = frame.copy()
        for d in detections:
            bbox = d.get("bbox") or (0, 0, 0, 0)
            x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2)
            text = str(d.get("text", ""))[:120]
            conf = float(d.get("confidence", 0.0))
            cv2.putText(canvas, f"{text} [{conf:.2f}]", (x0, max(15, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        for t in tracks:
            bbox = t.get("bbox") or (0, 0, 0, 0)
            x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
            cv2.putText(canvas, f"Track {t.get('tracking_id','?')}", (x0, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame: {frame_index}  Time: {timestamp:.6f}s",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        out = self.output_dir / f"frame_{int(frame_index):08d}.jpg"
        cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return out
