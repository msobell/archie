import base64
import io
import json
import os
import re
from pathlib import Path

import anthropic
import fitz
from PIL import Image

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

VISION_PROMPT = (
    "This is a scanned page of an HOA (Homeowners Association) governing document "
    "(bylaws, CC&Rs, or architectural design guidelines). "
    "Transcribe the text exactly as written, preserving line breaks and section headings. "
    "Output only the transcribed text with no commentary."
)

DIGIT_THRESHOLD = 0.06
SUSPICIOUS_THRESHOLD = 0.07


def page_to_image(pdf_path: Path, page_num: int, dpi: int = 300) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def claude_vision_ocr(img: Image.Image, client: anthropic.Anthropic) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    )
    return response.content[0].text.strip()


def tesseract_ocr(img: Image.Image) -> str:
    return pytesseract.image_to_string(img)


def compute_scores(text: str) -> dict:
    words = text.split()
    if not words:
        return {"digit_ratio": 0.0, "suspicious_ratio": 0.0, "word_count": 0}
    digit_ratio = sum(1 for w in words if re.fullmatch(r"\d+", w)) / len(words)
    suspicious_ratio = sum(
        1 for w in words if re.search(r'[^a-zA-Z\s\'\-\.,;:!?"\(\)]', w)
    ) / len(words)
    return {
        "digit_ratio": round(digit_ratio, 4),
        "suspicious_ratio": round(suspicious_ratio, 4),
        "word_count": len(words),
    }


def is_bad_ocr(scores: dict) -> bool:
    if scores["word_count"] < 20:
        return False
    return scores["digit_ratio"] > DIGIT_THRESHOLD or scores["suspicious_ratio"] > SUSPICIOUS_THRESHOLD


def _load_index(index_path: Path) -> dict:
    if index_path.exists():
        return {f"{e['doc_id']}_{e['page']:04d}": e
                for e in json.loads(index_path.read_text())}
    return {}


def _save_index(index: dict, index_path: Path):
    entries = sorted(index.values(), key=lambda e: (e["doc_id"], e["page"]))
    index_path.write_text(json.dumps(entries, indent=2))


def convert_pdfs(source_dir: str, docs_dir: str):
    """
    For each PDF in source_dir, run Claude Vision OCR and write a combined .txt to docs_dir.
    Skips PDFs whose output .txt already exists. Uses a per-page index for resumability.
    """
    source_path = Path(source_dir)
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(source_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {source_dir}.")
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    index_path = source_path / "index.json"
    index = _load_index(index_path)

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        out_txt = docs_path / f"{doc_id}.txt"

        if out_txt.exists():
            print(f"  Skipping {pdf_path.name} (already converted).")
            continue

        print(f"  OCR-ing {pdf_path.name}...")
        doc = fitz.open(str(pdf_path))
        page_count = doc.page_count
        doc.close()

        page_texts = []
        for page_num in range(page_count):
            key = f"{doc_id}_{page_num:04d}"
            if key in index:
                page_texts.append(index[key]["text"])
                continue

            print(f"    Page {page_num + 1}/{page_count}...", end=" ", flush=True)
            img = page_to_image(pdf_path, page_num)
            used_tesseract = False
            try:
                text = claude_vision_ocr(img, client)
            except anthropic.BadRequestError:
                if _TESSERACT_AVAILABLE:
                    print(f"content filtered, falling back to Tesseract...", end=" ", flush=True)
                    try:
                        text = tesseract_ocr(img)
                        used_tesseract = True
                    except Exception as te:
                        print(f"Tesseract failed ({te}), skipping page")
                        text = ""
                else:
                    print(f"content filtered, no Tesseract fallback available, skipping page")
                    text = ""

            scores = compute_scores(text)

            if is_bad_ocr(scores):
                print(f"low quality (scores={scores})")
            else:
                print(f"ok ({scores['word_count']} words)")

            page_texts.append(text)
            index[key] = {
                "doc_id": doc_id,
                "page": page_num,
                "char_count": len(text),
                "flagged_blank": len(text) < 50,
                "used_vision": not used_tesseract,
                "scores": scores,
                "text": text,
            }
            _save_index(index, index_path)

        combined = "\n\n".join(page_texts)
        out_txt.write_text(combined, encoding="utf-8")
        print(f"  Wrote {out_txt.name} ({len(combined)} chars).")
