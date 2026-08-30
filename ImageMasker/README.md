# ImageMasker

ImageMasker is integrated into the unified DataVeil FastAPI application.

Run DataVeil from the repository root:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then open <http://127.0.0.1:8000/imagemasker>.

The web workflow accepts PNG, JPG, and JPEG files, segments people and creates
pixel-level semantic masks for faces locally, supports automatic and manual
region selection, and offers blur, pixelate, mosaic, black box, custom color,
noise, sticker, and silhouette effects.

The web service uses the bundled `yolov8n-seg.pt` file explicitly and forces
YOLO person inference onto the CPU. Face masks use the bundled BiSeNet
ResNet-18 semantic parser trained on CelebAMask-HQ. It does not invoke a
cloud API or persist uploads.

## Recovery workflow

Send only the masked PNG to the external AI tool. DataVeil retains the
recovery package temporarily in the current browser tab. Later, upload only
the edited result; the browser attaches the recovery package automatically.

The recovery package is sensitive. It includes the exact saved semantic or
instance mask, a masked alignment reference, and the original pixels inside
the protected regions. Restoration reuses this saved mask and never
re-segments the returned image.

The face parser and its bundled weights come from the MIT-licensed
`yakhyo/face-parsing` project. Its license is included at
`ImageMasker/models/FACE_PARSING_LICENSE.txt`.

## Preserved command-line tools

`unified_masker.py` and `unmasker.py` remain available for legacy CLI use.
The optional InsightFace CLI path still requires `insightface` and
`onnxruntime` plus a locally available InsightFace model. The FastAPI
application does not require or download that model.
