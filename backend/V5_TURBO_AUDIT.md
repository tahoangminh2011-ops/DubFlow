# VideoOCR V5 Turbo audit

## User-run root cause evidence
The supplied Windows log shows:
- ONNX Runtime tried to load CUDA but failed because `cublasLt64_12.dll` was missing.
- RapidOCR then explicitly shifted inference to `CPUExecutionProvider`.
- Backend reported `gpu_active=False` / `providers=[]`.
- Fast scan remained at `candidates=0`.
- At 16.7% progress it had scanned 48,480 samples, so the previous run could not meet the 3–5 minute one-hour target.

These observations are from the user's captured run, not a synthetic benchmark.

## V5 changes
- Preload CUDA/cuDNN DLLs from NVIDIA Python packages/CUDA_PATH before ORT/RapidOCR.
- Fail early in target mode when an NVIDIA GPU is present but OCR cannot activate CUDA.
- Default fast scan: 20 FPS at 224px, retaining 0.1s temporal coverage.
- Optional NVDEC + `scale_cuda` path when FFmpeg exposes it.
- More recall-tolerant fast presence gate; candidate generator remains non-OCR.
- Existing OCR/tracking/timeline architecture remains event-driven.
- JSON remains master; SRT derives from final JSON.

## Local synthetic checks
- 30s synthetic clip fast scan: 1.848s, ~16.2x realtime.
- 11 candidate windows produced, including short 0.2s caption transitions.
- Unit/integration tests: 22 passed.

The 60-minute 3–5 minute full-OCR target is not claimed as universally achieved until the user's Windows/NVIDIA runtime passes the GPU doctor and an actual full-video benchmark.
