# ImageMasker (Beginner-Friendly Guide)

This project lets you automatically hide (mask) a person’s **face**, **body**, or **both** in a photo.  
It also includes a helper script to **restore only the masked area** after you edit the photo with an AI tool.

You do **not** need AI or computer vision experience to run this.

---

## What You’ll Get

Running the main script creates:
- `out/masked.png`: the final image with blur/pixelate/black mask applied
- `out/chosen_mask.png`: the mask used
- `out/overlay.png`: a visual overlay showing detected people and faces (for debugging)

Optional AI-prep exports:
- `out/masked_for_ai.png`
- `out/mask_for_ai.png`
- `out/transform.json`

---

## 1) Install Python (Windows + macOS)

You need **Python 3.10+**.

### Windows
1. Download Python from the official site.
2. During install, **check the box**: “Add Python to PATH”.
3. Open **PowerShell** and run:
```powershell
python --version
```
If you see a version number (like `Python 3.11.x`), you’re good.

### macOS
Open **Terminal** and run:
```bash
python3 --version
```
If it says `Python 3.10+`, you’re good.  
If not, install Python from the official site or via Homebrew:
```bash
brew install python
```

---

## 2) Open This Folder

Open a terminal in this folder:

`e:\Valluvan\localspider\ValluvanTestSpace\ImagetoTest\ImageMasker`

---

## 3) Create a Virtual Environment (Recommended)

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS (Terminal)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` appear in your terminal after activation.

---

## 4) Install Required Libraries

### Windows
```powershell
python -m pip install --upgrade pip
pip install ultralytics insightface onnxruntime opencv-python numpy
```

### macOS
```bash
python3 -m pip install --upgrade pip
pip install ultralytics insightface onnxruntime opencv-python numpy
```

Optional (GPU on Windows with NVIDIA):
```powershell
pip install onnxruntime-gpu
```

---

## 5) Run the Main Script

There is a sample image named `TestImage.png`.

### Example (mask the middle person’s face with blur)
```powershell
python unified_masker.py --image TestImage.png --choose middle --mask face --effect blur --outdir out
```

### Example (mask the left person’s body with pixelation)
```powershell
python unified_masker.py --image TestImage.png --choose left --mask body --effect pixelate --outdir out
```

---

## 6) Choose Which Person to Mask

The script labels people from **left → right**:
- If there are **3 people**, labels are: `left`, `middle`, `right`
- Otherwise, labels are: `p0`, `p1`, `p2`, ...

You can also use numeric index:
```powershell
--choose 0
```

---

## 7) Optional: Prepare for AI Editing

This exports the image + mask in a square format with metadata:
```powershell
python unified_masker.py --image TestImage.png --choose middle --mask both --effect blur --prepare_for_ai --outdir out
```

You’ll get:
- `out/masked_for_ai.png`
- `out/mask_for_ai.png`
- `out/transform.json`

---

## 8) Restore Masked Area After AI Editing (Optional)

If you edit the masked image with another tool, you can restore only the masked region:

```powershell
python unmasker.py --edited ai_output.png --original TestImage.png --mask out\chosen_mask.png --meta out\transform.json --outdir out_unmask --outname final_unmasked.png
```

This will:
- Paste back the original pixels **only inside the mask**
- Keep your AI edits outside the masked area

---

## Common Issues

**1) “No persons detected”**
- Try a clearer image
- Lower confidence:
```powershell
--conf 0.25
```

**2) InsightFace install fails**
- Upgrade pip:
```powershell
python -m pip install --upgrade pip
```
- On Windows, install **Microsoft C++ Build Tools** if required

**3) Slow performance**
- Use a smaller image
- GPU helps (Windows + NVIDIA): `pip install onnxruntime-gpu`

---

## Quick Reference

```powershell
python unified_masker.py --image <IMAGE> --choose <left|middle|right|p0|p1|...> --mask <face|body|both> --effect <blur|pixelate|black> --outdir out
```

---

If you want, I can also add a one-click `run.bat` / `run.command` launcher.
