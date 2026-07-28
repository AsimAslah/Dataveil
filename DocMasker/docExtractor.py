#!/usr/bin/env python3
"""
Convert documents in a folder (PDF/DOCX/PPTX) into Markdown files.

- Outputs to a subfolder named "Markdown" inside the input folder
- Mirrors the input folder structure
- Tries to use Docling if installed; falls back to standard parsers
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

SUPPORTED_EXTS = {".pdf", ".docx", ".pptx"}


def safe_read_text(path: Path) -> str:
    # for any small text reads if needed; not used for binary docs
    return path.read_text(encoding="utf-8", errors="replace")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_output_path(input_root: Path, output_root: Path, src_file: Path) -> Path:
    rel = src_file.relative_to(input_root)
    return output_root / rel.with_suffix(".md")


# -----------------------------
# Docling (preferred) converter
# -----------------------------
def try_docling_convert(src_file: Path) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Returns: (ok, markdown, error_message)
    """
    try:
        # Docling API may vary by version; this tries a common usage pattern,
        # and falls back gracefully if the import or API doesn't match.
        from docling.document_converter import DocumentConverter  # type: ignore
    except Exception as e:
        return False, None, f"Docling not available: {e}"

    try:
        converter = DocumentConverter()
        result = converter.convert(str(src_file))
        doc = result.document

        # Newer docling versions often have export methods like:
        # - doc.export_to_markdown()
        # - doc.to_markdown()
        # We'll try a few safely.
        for attr in ("export_to_markdown", "to_markdown", "as_markdown"):
            if hasattr(doc, attr):
                md = getattr(doc, attr)()
                if isinstance(md, str) and md.strip():
                    return True, md, None

        # Some versions put markdown into result or allow exporting via result
        for obj in (result, doc):
            for attr in ("markdown", "md"):
                if hasattr(obj, attr):
                    md = getattr(obj, attr)
                    if isinstance(md, str) and md.strip():
                        return True, md, None

        return False, None, "Docling conversion succeeded but no markdown export method found."
    except Exception as e:
        return False, None, f"Docling failed: {e}"


# -----------------------------
# Fallback converters
# -----------------------------
def docx_to_markdown(src_file: Path) -> str:
    from docx import Document  # type: ignore

    doc = Document(str(src_file))
    lines = []

    # Basic paragraph extraction
    for p in doc.paragraphs:
        text = (p.text or "").rstrip()
        if not text:
            lines.append("")
            continue
        lines.append(text)

    # Tables (simple)
    for t_i, table in enumerate(doc.tables, start=1):
        lines.append("")
        lines.append(f"## Table {t_i}")
        rows = []
        for row in table.rows:
            rows.append([cell.text.strip().replace("\n", " ") for cell in row.cells])

        if rows:
            header = rows[0]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            for r in rows[1:]:
                lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines).strip() + "\n"


def pptx_to_markdown(src_file: Path) -> str:
    from pptx import Presentation  # type: ignore

    prs = Presentation(str(src_file))
    lines = []
    for i, slide in enumerate(prs.slides, start=1):
        title = None
        # Attempt to find a title shape
        for shape in slide.shapes:
            if hasattr(shape, "is_placeholder") and shape.is_placeholder:
                try:
                    if shape.placeholder_format.type == 1:  # TITLE
                        title = (shape.text or "").strip()
                except Exception:
                    pass
        if not title:
            title = f"Slide {i}"

        lines.append(f"# {title}")
        slide_texts = []
        for shape in slide.shapes:
            if not hasattr(shape, "text"):
                continue
            txt = (shape.text or "").strip()
            if not txt:
                continue
            # avoid duplicating the title
            if txt == title:
                continue
            slide_texts.append(txt)

        if slide_texts:
            for block in slide_texts:
                # Split into bullet-ish lines
                for ln in block.splitlines():
                    ln = ln.strip()
                    if ln:
                        lines.append(f"- {ln}")
        else:
            lines.append("_No text found on this slide._")

        lines.append("")  # blank line between slides

    return "\n".join(lines).strip() + "\n"


def pdf_to_markdown(src_file: Path) -> str:
    """
    Text extraction fallback for PDFs. For best quality use Docling.
    """
    import fitz  # PyMuPDF  # type: ignore

    doc = fitz.open(str(src_file))
    lines = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text = page.get_text("text").strip()
        lines.append(f"## Page {pno + 1}")
        lines.append("")
        lines.append(text if text else "_No extractable text on this page (may be scanned/image-based)._")
        lines.append("\n---\n")
    return "\n".join(lines).strip() + "\n"


def convert_one(src_file: Path, prefer_docling: bool = True) -> Tuple[str, str]:
    """
    Returns: (markdown, method_used)
    """
    ext = src_file.suffix.lower()

    if prefer_docling:
        ok, md, err = try_docling_convert(src_file)
        if ok and md:
            return md, "docling"
        # if docling is present but fails, we fall back
        # (you could log err if you want)

    if ext == ".docx":
        return docx_to_markdown(src_file), "python-docx"
    if ext == ".pptx":
        return pptx_to_markdown(src_file), "python-pptx"
    if ext == ".pdf":
        return pdf_to_markdown(src_file), "pymupdf"

    raise ValueError(f"Unsupported file type: {src_file}")


def iter_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert PDF/DOCX/PPTX in a folder to Markdown.")
    ap.add_argument("folder", type=str, help="Input folder containing documents.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--no-docling", action="store_true", help="Disable Docling and use fallbacks only.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .md outputs.")
    ap.add_argument(
        "--dataveil-json-file",
        type=str,
        default=None,
        help="Optional path for structured DataVeil runtime JSON output."
    )
    args = ap.parse_args()

    input_root = Path(args.folder).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {input_root}")

    output_root = input_root / "Markdown"
    ensure_dir(output_root)

    files = iter_files(input_root, recursive=args.recursive)

    # Avoid converting anything inside the output folder itself
    files = [f for f in files if output_root not in f.parents]

    if not files:
        print("No supported files found (.pdf, .docx, .pptx).")
        return 0

    prefer_docling = not args.no_docling

    converted = 0
    skipped = 0

    for src in files:
        out_path = rel_output_path(input_root, output_root, src)
        ensure_dir(out_path.parent)

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            md, method = convert_one(src, prefer_docling=prefer_docling)
            header = f"<!-- Source: {src.relative_to(input_root)} | Method: {method} -->\n\n"
            out_path.write_text(header + md, encoding="utf-8")
            converted += 1
            print(f"[OK] {src} -> {out_path} ({method})")
        except Exception as e:
            print(f"[FAIL] {src}: {e}")

    if args.dataveil_json_file:
        payload = {
            "inputFolder": str(input_root),
            "outputFolder": str(output_root),
            "converted": converted,
            "skipped": skipped
        }
        json_output_path = Path(args.dataveil_json_file).expanduser().resolve()
        ensure_dir(json_output_path.parent)
        json_output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Converted: {converted}, Skipped: {skipped}, Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
