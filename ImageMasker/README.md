# ImageMasker

ImageMasker is integrated into the unified DataVeil FastAPI application.

Run DataVeil from the repository root:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then open <http://127.0.0.1:8000/imagemasker>.

The web workflow accepts PNG, JPG, and JPEG files, detects people and faces
locally, supports automatic and manual region selection, and offers blur,
pixelate, mosaic, black box, custom color, noise, sticker, and silhouette
effects.

The web service uses the bundled `yolov8n-seg.pt` file explicitly and forces
YOLO inference onto the CPU. It does not invoke a cloud API or persist
uploads.

## Recovery workflow

Download both the masked PNG and DataVeil recovery JSON. Send only the masked
PNG to the external AI tool. Later, upload its edited result and the recovery
JSON to restore the protected pixels.

The recovery JSON is sensitive. It includes the selected mask, a masked
alignment reference, and the original pixels inside the protected regions.

## Preserved command-line tools

`unified_masker.py` and `unmasker.py` remain available for legacy CLI use.
The optional InsightFace CLI path still requires `insightface` and
`onnxruntime` plus a locally available InsightFace model. The FastAPI
application does not require or download that model.
