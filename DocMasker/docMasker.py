#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse

import spacy

# Phone validation
import phonenumbers
from phonenumbers import NumberParseException


# ----------------------------
# Markdown-safe helpers
# ----------------------------
FENCE_RE = re.compile(r"^\s*```")
INLINE_CODE_RE = re.compile(r"`[^`]*`")

def split_keep_inline_code(line: str) -> List[Tuple[str, bool]]:
    parts: List[Tuple[str, bool]] = []
    last = 0
    for m in INLINE_CODE_RE.finditer(line):
        if m.start() > last:
            parts.append((line[last:m.start()], False))
        parts.append((m.group(0), True))
        last = m.end()
    if last < len(line):
        parts.append((line[last:], False))
    return parts


# ----------------------------
# Typed mask mapping
# ----------------------------
# canonical_value -> typed mask token (e.g., "minhaj falaki" -> "Person1")
value_to_mask: Dict[str, str] = {}
# category -> next integer counter
category_counts: Dict[str, int] = {}

def next_category_mask(category: str) -> str:
    category_counts[category] = category_counts.get(category, 0) + 1
    return f"{category}{category_counts[category]}"

def ensure_typed_mask(key: str, category: str) -> str:
    """
    First-category-wins behavior:
    If the same key appears later with another category, we keep the first assigned mask.
    """
    if key not in value_to_mask:
        value_to_mask[key] = next_category_mask(category)
    return value_to_mask[key]


# ----------------------------
# PII patterns
# ----------------------------
EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE
)

# Phone candidate detector (loose on purpose; validation decides)
PHONE_CANDIDATE_RE = re.compile(
    r"""
    (?<!\w)
    (?:\+|00)?\d[\d\s().-]{6,}\d
    (?!\w)
    """,
    re.VERBOSE
)

# URL candidate detector:
# - http(s)://...
# - www....
# - bare domains like linkedin.com/in/...
URL_CANDIDATE_RE = re.compile(
    r"""
    (?<!@)                              # avoid starting inside an email local part
    (?:
        https?://[^\s<>()\[\]{}"']+     # scheme URLs
        |
        www\.[^\s<>()\[\]{}"']+         # www URLs
        |
        (?:[a-z0-9-]+\.)+[a-z]{2,}      # bare domain
            (?:/[^\s<>()\[\]{}"']*)?    # optional path
    )
    """,
    re.IGNORECASE | re.VERBOSE
)

# Social handle patterns (non-URL)
AT_HANDLE_RE = re.compile(r"(?<![\w.])@([A-Za-z0-9_][A-Za-z0-9_.-]{1,29})\b")

LABELED_HANDLE_RE = re.compile(
    r"(?i)\b(linkedin|github|twitter|x|instagram|ig|facebook|fb|telegram|tiktok|reddit)\b\s*[:\-]\s*([A-Za-z0-9_][A-Za-z0-9_.-]{1,49})\b"
)

TRAILING_URL_PUNCT = ".,;:!?)\u201d\u2019]}>\"'"

def normalize_key(s: str, case_insensitive: bool = True) -> str:
    s = s.strip()
    return s.lower() if case_insensitive else s


# ----------------------------
# Phone normalization + validation
# ----------------------------
def normalize_phone(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"[^\d+]", "", raw)
    if raw.startswith("00"):
        raw = "+" + raw[2:]
    return raw

def _valid_e164_or_international(raw_norm: str) -> bool:
    try:
        num = phonenumbers.parse(raw_norm, None)
        return phonenumbers.is_valid_number(num)
    except NumberParseException:
        return False

def _valid_with_region(raw_norm: str, region: str) -> bool:
    try:
        num = phonenumbers.parse(raw_norm, region)
        return phonenumbers.is_valid_number(num)
    except NumberParseException:
        return False

def is_valid_phone(raw: str, regions: Tuple[str, ...]) -> bool:
    raw_norm = normalize_phone(raw)

    digits = re.sub(r"\D", "", raw_norm)
    if len(digits) < 7:
        return False

    if raw_norm.startswith("+"):
        return _valid_e164_or_international(raw_norm)

    for r in regions:
        if _valid_with_region(raw_norm, r):
            return True

    return False

def detect_phone_spans(text: str, regions: Tuple[str, ...]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for m in PHONE_CANDIDATE_RE.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 7:
            continue
        if is_valid_phone(raw, regions):
            spans.append((m.start(), m.end(), raw))
    return spans


# ----------------------------
# URL normalization + classification
# ----------------------------
def clean_url(raw: str) -> str:
    u = raw.strip()
    while u and u[-1] in TRAILING_URL_PUNCT:
        u = u[:-1]
    u = u.strip("()[]{}<>")
    return u

def _parse_host(url: str) -> Optional[str]:
    u = clean_url(url)
    if u.lower().startswith("www."):
        u_for_parse = "http://" + u
    elif re.match(r"^https?://", u, re.IGNORECASE):
        u_for_parse = u
    else:
        u_for_parse = "http://" + u
    try:
        parsed = urlparse(u_for_parse)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None

def classify_url(url: str, social_domains: Tuple[str, ...]) -> str:
    """
    Returns 'social' or 'normal'
    """
    host = _parse_host(url)
    if not host:
        return "normal"
    for d in social_domains:
        d = d.lower()
        if host == d or host.endswith("." + d):
            return "social"
    return "normal"

def normalize_url_key(url: str) -> str:
    return clean_url(url).strip().lower()

def detect_url_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for m in URL_CANDIDATE_RE.finditer(text):
        raw = m.group(0)
        cleaned = clean_url(raw)
        if not cleaned:
            continue
        if "." not in cleaned:
            continue
        spans.append((m.start(), m.start() + len(cleaned), cleaned))
    return spans


# ----------------------------
# Social handle detection
# ----------------------------
def detect_social_handle_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []

    for m in AT_HANDLE_RE.finditer(text):
        raw = m.group(0)  # includes '@'
        spans.append((m.start(), m.end(), raw))

    for m in LABELED_HANDLE_RE.finditer(text):
        handle = m.group(2)
        spans.append((m.start(2), m.end(2), handle))

    return spans


# ----------------------------
# NER label -> category
# ----------------------------
def ner_label_to_category(label: str) -> str:
    if label == "PERSON":
        return "Person"
    if label == "ORG":
        return "Company"
    if label in ("GPE", "LOC", "FAC"):
        return "Place"
    return "Entity"


# ----------------------------
# Core masking logic (PHONE + EMAIL + URL + HANDLES + NER)
# ----------------------------
def mask_text_segment(
    nlp,
    text: str,
    case_insensitive: bool = True,
    entity_labels: Tuple[str, ...] = ("PERSON", "ORG", "GPE", "LOC"),
    mask_emails: bool = True,
    mask_phones: bool = True,
    phone_regions: Tuple[str, ...] = ("IN", "US", "GB", "AE", "AU", "CA"),
    # URL/social controls
    mask_social_urls: bool = True,
    mask_normal_urls: bool = False,
    mask_social_handles: bool = True,
    social_domains: Tuple[str, ...] = (
        "linkedin.com",
        "x.com",
        "twitter.com",
        "instagram.com",
        "facebook.com",
        "t.me",
        "telegram.me",
        "wa.me",
        "whatsapp.com",
        "tiktok.com",
        "reddit.com",
        "threads.net",
        # Add "github.com" if you want to treat it as social/private:
        # "github.com",
    ),
) -> Tuple[str, List[Dict[str, object]]]:
    if not text.strip():
        return text, []

    # spans: (start, end, replacement, category, sourceText)
    spans: List[Tuple[int, int, str, str, str]] = []

    def overlaps_existing(s: int, e: int) -> bool:
        return any(not (e <= s2 or s >= e2) for s2, e2, _, _, _ in spans)

    # 0) PHONE spans
    if mask_phones:
        for start, end, raw in detect_phone_spans(text, phone_regions):
            key = normalize_phone(raw)  # canonical
            rep = ensure_typed_mask(key, "Phone")
            spans.append((start, end, rep, "Phone", raw))

    # 1) EMAIL spans
    if mask_emails:
        for m in EMAIL_RE.finditer(text):
            raw = m.group(0)
            key = normalize_key(raw, case_insensitive=True)  # always case-insensitive for emails
            rep = ensure_typed_mask(key, "MailID")
            spans.append((m.start(), m.end(), rep, "MailID", raw))

    # 2) URL spans
    for start, end, url in detect_url_spans(text):
        if overlaps_existing(start, end):
            continue

        kind = classify_url(url, social_domains)
        if kind == "social":
            if not mask_social_urls:
                continue
            key = normalize_url_key(url)
            rep = ensure_typed_mask(key, "SocialMediaID")
            category = "SocialMediaID"
        else:
            if not mask_normal_urls:
                continue
            key = normalize_url_key(url)
            rep = ensure_typed_mask(key, "URL")
            category = "URL"

        spans.append((start, end, rep, category, url))

    # 3) Social handles (non-URL)
    if mask_social_handles:
        for start, end, raw in detect_social_handle_spans(text):
            if overlaps_existing(start, end):
                continue
            key = normalize_key(raw, case_insensitive=True)  # handles are case-insensitive
            rep = ensure_typed_mask(key, "SocialMediaID")
            spans.append((start, end, rep, "SocialMediaID", raw))

    # 4) NER entities
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ not in entity_labels:
            continue

        ent_start = ent.start_char
        ent_end = ent.end_char
        if overlaps_existing(ent_start, ent_end):
            continue

        raw = ent.text.strip()
        if not raw:
            continue

        category = ner_label_to_category(ent.label_)
        key = normalize_key(raw, case_insensitive=case_insensitive)
        rep = ensure_typed_mask(key, category)
        spans.append((ent_start, ent_end, rep, category, raw))

    if not spans:
        return text, []

    ordered = sorted(spans, key=lambda x: x[0])
    token_payloads: List[Dict[str, object]] = []
    for start, end, replacement, category, source_text in ordered:
        token_payloads.append({
            "category": category,
            "sourceText": source_text,
            "placeholder": replacement,
            "spanStart": start,
            "spanEnd": end
        })

    # apply replacements from end -> start
    spans.sort(key=lambda x: x[0], reverse=True)
    out = text
    for start, end, rep, _, _ in spans:
        out = out[:start] + rep + out[end:]
    return out, token_payloads


def mask_markdown_file(
    nlp,
    src_path: Path,
    dst_path: Path,
    case_insensitive: bool = True,
    entity_labels: Tuple[str, ...] = ("PERSON", "ORG", "GPE", "LOC"),
    mask_emails: bool = True,
    mask_phones: bool = True,
    phone_regions: Tuple[str, ...] = ("IN", "US", "GB", "AE", "AU", "CA"),
    mask_social_urls: bool = True,
    mask_normal_urls: bool = False,
    mask_social_handles: bool = True,
    social_domains: Tuple[str, ...] = (
        "linkedin.com",
        "x.com",
        "twitter.com",
        "instagram.com",
        "facebook.com",
        "t.me",
        "telegram.me",
        "wa.me",
        "whatsapp.com",
        "tiktok.com",
        "reddit.com",
        "threads.net",
    ),
) -> Dict[str, object]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    source_text = src_path.read_text(encoding="utf-8", errors="replace")
    in_fenced_block = False
    out_lines: List[str] = []
    token_payloads: List[Dict[str, object]] = []
    offset = 0

    for line in source_text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        if FENCE_RE.match(line):
            in_fenced_block = not in_fenced_block
            out_lines.append(line)
            continue

        if in_fenced_block:
            out_lines.append(line)
            continue

        segments = split_keep_inline_code(line)
        masked_parts: List[str] = []
        segment_offset = 0
        for seg_text, protected in segments:
            absolute_segment_start = line_start + segment_offset
            segment_offset += len(seg_text)
            if protected:
                masked_parts.append(seg_text)
            else:
                masked_segment, segment_tokens = mask_text_segment(
                    nlp=nlp,
                    text=seg_text,
                    case_insensitive=case_insensitive,
                    entity_labels=entity_labels,
                    mask_emails=mask_emails,
                    mask_phones=mask_phones,
                    phone_regions=phone_regions,
                    mask_social_urls=mask_social_urls,
                    mask_normal_urls=mask_normal_urls,
                    mask_social_handles=mask_social_handles,
                    social_domains=social_domains,
                )
                masked_parts.append(masked_segment)
                for token in segment_tokens:
                    token_payloads.append({
                        "category": token["category"],
                        "sourceText": token["sourceText"],
                        "placeholder": token["placeholder"],
                        "spanStart": int(token["spanStart"]) + absolute_segment_start,
                        "spanEnd": int(token["spanEnd"]) + absolute_segment_start
                    })

        out_lines.append("".join(masked_parts))

    masked_text = "".join(out_lines)
    dst_path.write_text(masked_text, encoding="utf-8")

    return {
        "sourcePath": str(src_path),
        "maskedPath": str(dst_path),
        "sourceText": source_text,
        "maskedText": masked_text,
        "tokens": token_payloads
    }


def iter_md_files(root: Path, recursive: bool = True) -> List[Path]:
    return [p for p in (root.rglob("*.md") if recursive else root.glob("*.md")) if p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mask entities + emails + phones + (optionally) URLs/socials in Markdown using spaCy, with typed masks (Person1, Place1...)."
    )
    ap.add_argument("input_folder", type=str, help="Folder containing Markdown files.")
    ap.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder. Default: sibling folder named 'MaskedMarkdown' next to input folder.",
    )
    ap.add_argument("--non-recursive", action="store_true", help="Do not traverse subfolders.")
    ap.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Treat 'California' and 'california' as different keys for NER entities (not recommended).",
    )
    ap.add_argument("--model", type=str, default="en_core_web_trf", help="spaCy model name.")
    ap.add_argument(
        "--labels",
        type=str,
        default="PERSON,ORG,GPE,LOC,FAC",
        help="Comma-separated spaCy entity labels to mask. Default: PERSON,ORG,GPE,LOC,FAC",
    )

    ap.add_argument("--no-email", action="store_true", help="Disable email masking.")
    ap.add_argument("--no-phone", action="store_true", help="Disable phone number masking.")
    ap.add_argument(
        "--phone-regions",
        type=str,
        default="IN,US,GB,AE,AU,CA",
        help="Comma-separated region codes to try when phone numbers do not include '+' (fallback parsing).",
    )

    # URL/social flags (defaults: mask socials, keep normal URLs for scraping)
    ap.add_argument("--no-mask-social-urls", action="store_true", help="Do NOT mask social-media URLs.")
    ap.add_argument("--mask-normal-urls", action="store_true", default=False, help="Mask non-social URLs (default: OFF).")
    ap.add_argument("--no-mask-social-handles", action="store_true", help="Do NOT mask social handles.")

    ap.add_argument(
        "--social-domains",
        type=str,
        default="linkedin.com,x.com,twitter.com,instagram.com,facebook.com,t.me,telegram.me,wa.me,whatsapp.com,tiktok.com,reddit.com,threads.net",
        help="Comma-separated domains treated as social.",
    )
    ap.add_argument(
        "--dataveil-json-file",
        type=str,
        default=None,
        help="Optional path for structured DataVeil runtime JSON output."
    )

    args = ap.parse_args()

    in_root = Path(args.input_folder).expanduser().resolve()
    if not in_root.exists() or not in_root.is_dir():
        raise SystemExit(f"Input folder not found: {in_root}")

    out_root = Path(args.output_folder).expanduser().resolve() if args.output_folder else in_root.parent / "MaskedMarkdown"
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        nlp = spacy.load(args.model)
    except Exception as e:
        raise SystemExit(
            f"Failed to load spaCy model '{args.model}'. "
            f"Try: python -m spacy download {args.model}\nError: {e}"
        )

    entity_labels = tuple([x.strip() for x in args.labels.split(",") if x.strip()])
    phone_regions = tuple([x.strip().upper() for x in args.phone_regions.split(",") if x.strip()])
    social_domains = tuple([x.strip().lower() for x in args.social_domains.split(",") if x.strip()])

    mask_social_urls = not args.no_mask_social_urls
    mask_social_handles = not args.no_mask_social_handles
    mask_normal_urls = args.mask_normal_urls

    recursive = not args.non_recursive
    md_files = iter_md_files(in_root, recursive=recursive)

    if not md_files:
        print("No .md files found.")
        return 0

    # Reset globals for a fresh run
    value_to_mask.clear()
    category_counts.clear()
    file_payloads: List[Dict[str, object]] = []

    for src in md_files:
        rel = src.relative_to(in_root)
        dst = out_root / rel
        file_payload = mask_markdown_file(
            nlp=nlp,
            src_path=src,
            dst_path=dst,
            case_insensitive=(not args.case_sensitive),
            entity_labels=entity_labels,
            mask_emails=(not args.no_email),
            mask_phones=(not args.no_phone),
            phone_regions=phone_regions,
            mask_social_urls=mask_social_urls,
            mask_normal_urls=mask_normal_urls,
            mask_social_handles=mask_social_handles,
            social_domains=social_domains,
        )
        file_payloads.append(file_payload)
        print(f"[OK] {src} -> {dst}")

    # Print and save mapping (canonical_value -> typed_mask)
    mapping_lines = [f"{k}-{v}" for k, v in sorted(value_to_mask.items(), key=lambda kv: kv[0])]
    print("\n=== VALUE -> MASK MAP ===")
    for line in mapping_lines:
        print(line)

    map_path = out_root / "_value_mask_map.txt"
    map_path.write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")
    print(f"\nSaved mapping to: {map_path}")

    # Also save category counters for quick inspection
    counts_path = out_root / "_mask_category_counts.txt"
    counts_lines = [f"{k}:{v}" for k, v in sorted(category_counts.items(), key=lambda kv: kv[0])]
    counts_path.write_text("\n".join(counts_lines) + "\n", encoding="utf-8")

    if args.dataveil_json_file:
        runtime_payload = {
            "files": file_payloads
        }
        runtime_path = Path(args.dataveil_json_file).expanduser().resolve()
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(runtime_payload, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python docMasker.py E:\Valluvan\DataVeil\DocMasker\TestFolder\Markdown
