import uuid
from pathlib import Path
from config import SUPPORTED_EXTENSIONS


def extract(file_path: str, image_understanding: bool = False) -> dict:
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    parsers = {
        ".pdf":  lambda p: _parse_pdf(p, image_understanding),
        ".docx": lambda p: _parse_docx(p, image_understanding),
        ".pptx": lambda p: _parse_pptx(p, image_understanding),
        ".xlsx": _parse_xlsx,
        ".csv":  _parse_csv,
        ".txt":  _parse_txt,
        ".png":  lambda p: _parse_image(p, image_understanding),
        ".jpg":  lambda p: _parse_image(p, image_understanding),
        ".jpeg": lambda p: _parse_image(p, image_understanding),
    }

    result = parsers[ext](path)

    if isinstance(result, tuple):
        blocks, image_map = result
    else:
        blocks    = result
        image_map = {}

    return {
        "doc_id":    str(uuid.uuid4()),
        "filename":  path.name,
        "ext":       ext,
        "blocks":    blocks,
        "full_text": _build_full_text(blocks),
        "image_map": image_map,
    }


# ── PNG normaliser ────────────────────────────────────────────────────────────

def _to_png(image_bytes: bytes) -> bytes:
    """Convert any image format to PNG bytes for consistent storage."""
    try:
        from PIL import Image
        import io
        img    = Image.open(io.BytesIO(image_bytes))
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()
    except Exception:
        return image_bytes


# ── Image understanding with GPT-4.1-mini ─────────────────────────────────────

def _describe_image_with_llm(image_bytes: bytes, context: str = "") -> str:
    """Send image bytes to GPT-4.1-mini vision and return a text description."""
    import base64
    from openai import AzureOpenAI
    from config import (
        AZURE_OPENAI_LLM_KEY,
        AZURE_LLM_ENDPOINT,
        AZURE_LLM_API_VERSION,
        AZURE_LLM_DEPLOYMENT,
    )

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_LLM_KEY,
        azure_endpoint=AZURE_LLM_ENDPOINT,
        api_version=AZURE_LLM_API_VERSION,
    )

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "Describe this image in detail for document search purposes. "
        "If it contains a chart or graph, describe the data, trends, and key values. "
        "If it contains a table, extract the data as plain text. "
        "If it contains a diagram or flowchart, explain what it shows. "
        "If it contains a photo, describe what is visible. "
        "Be specific with any numbers, labels, or text you can read."
    )
    if context:
        prompt += f" Context: {context}"

    try:
        response = client.chat.completions.create(
            model=AZURE_LLM_DEPLOYMENT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {
                            "url":    f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[vision] image description failed: {e}")
        return ""


# ── PDF ───────────────────────────────────────────────────────────────────────

def _parse_pdf(path: Path, image_understanding: bool = False) -> tuple:
    import fitz
    from rapidocr_onnxruntime import RapidOCR

    doc         = fitz.open(str(path))
    blocks      = []
    image_map   = {}
    char_cursor = 0

    # ── pass 1: normal text extraction ───────────────────────────────────
    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue

            lines = []
            for line in block["lines"]:
                line_text = " ".join(
                    span["text"] for span in line["spans"]
                ).strip()
                if line_text:
                    lines.append(line_text)

            text = "\n".join(lines).strip()
            if not text:
                continue

            first_size = block["lines"][0]["spans"][0]["size"] \
                if block["lines"] else 0
            elem_type  = "heading" if first_size >= 14 else "paragraph"

            start = char_cursor
            end   = char_cursor + len(text)
            char_cursor = end + 1

            blocks.append({
                "type":       elem_type,
                "text":       text,
                "page":       page_num,
                "start_char": start,
                "end_char":   end,
                "has_image":  False,
            })

        # ── embedded images (toggle ON only) ──────────────────────────────
        if image_understanding:
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref        = img[0]
                    base_image  = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    if len(image_bytes) < 5000:
                        continue

                    print(
                        f"[vision] PDF page {page_num} "
                        f"image {img_index+1}/{len(image_list)}..."
                    )
                    description = _describe_image_with_llm(
                        image_bytes,
                        context=f"page {page_num} of PDF document"
                    )

                    if description:
                        block_index = len(blocks)
                        start       = char_cursor
                        end         = char_cursor + len(description)
                        char_cursor = end + 1

                        blocks.append({
                            "type":       "paragraph",
                            "text":       f"[Image on page {page_num}]: {description}",
                            "page":       page_num,
                            "start_char": start,
                            "end_char":   end,
                            "has_image":  True,
                        })
                        image_map[block_index] = _to_png(image_bytes)   # ← normalised

                except Exception as e:
                    print(f"[vision] failed to process image: {e}")
                    continue

    doc.close()

    # ── check if text extraction produced meaningful content ──────────────
    full_text = _build_full_text(blocks)
    if len(full_text.strip()) >= 50:
        return blocks, image_map

    # ── pass 2: image-based PDF — RapidOCR fallback ───────────────────────
    print(f"[ocr] '{path.name}' has no text layer — running RapidOCR...")
    blocks      = []
    image_map   = {}
    char_cursor = 0
    ocr_engine  = RapidOCR()
    doc         = fitz.open(str(path))

    for page_num, page in enumerate(doc, start=1):
        print(f"[ocr] processing page {page_num}/{len(doc)}...")

        mat       = fitz.Matrix(2, 2)
        pix       = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        result, _ = ocr_engine(img_bytes)

        if not result:
            print(f"[ocr] page {page_num} — no text detected")
            continue

        result_sorted = sorted(result, key=lambda x: x[0][0][1])
        page_lines    = [
            item[1].strip()
            for item in result_sorted
            if item[1].strip() and item[2] > 0.5
        ]

        if not page_lines:
            continue

        page_text   = "\n".join(page_lines)
        block_index = len(blocks)
        start       = char_cursor
        end         = char_cursor + len(page_text)
        char_cursor = end + 1

        blocks.append({
            "type":       "paragraph",
            "text":       page_text,
            "page":       page_num,
            "start_char": start,
            "end_char":   end,
            "has_image":  True,
        })

        image_map[block_index] = _to_png(img_bytes)   # ← normalised

        print(f"[ocr] page {page_num} — extracted {len(page_lines)} lines")

    doc.close()

    full_text = _build_full_text(blocks)
    if len(full_text.strip()) < 50:
        raise ValueError(
            f"'{path.name}' could not be read — either the PDF is empty, "
            f"corrupted, or the image quality is too low for OCR."
        )

    print(f"[ocr] '{path.name}' complete — {len(blocks)} pages extracted")
    return blocks, image_map


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _parse_docx(path: Path, image_understanding: bool = False) -> tuple:
    from docx import Document

    doc         = Document(str(path))
    blocks      = []
    image_map   = {}
    char_cursor = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style     = para.style.name
        elem_type = "heading" if "heading" in style.lower() else "paragraph"

        start = char_cursor
        end   = char_cursor + len(text)
        char_cursor = end + 1

        blocks.append({
            "type":       elem_type,
            "text":       text,
            "page":       None,
            "style":      style,
            "start_char": start,
            "end_char":   end,
            "has_image":  False,
        })

    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        text = "\n".join(rows).strip()
        if not text:
            continue

        start = char_cursor
        end   = char_cursor + len(text)
        char_cursor = end + 1

        blocks.append({
            "type":       "table",
            "text":       text,
            "page":       None,
            "style":      "Table",
            "start_char": start,
            "end_char":   end,
            "has_image":  False,
        })

    # ── embedded images (toggle ON only) ──────────────────────────────────
    if image_understanding:
        for i, rel in enumerate(doc.part.rels.values()):
            if "image" not in rel.reltype:
                continue
            try:
                image_bytes = rel.target_part.blob

                if len(image_bytes) < 5000:
                    continue

                print(f"[vision] DOCX image {i+1}...")
                description = _describe_image_with_llm(
                    image_bytes,
                    context="embedded image in Word document"
                )

                if description:
                    block_index = len(blocks)
                    start       = char_cursor
                    end         = char_cursor + len(description)
                    char_cursor = end + 1

                    blocks.append({
                        "type":       "paragraph",
                        "text":       f"[Embedded image]: {description}",
                        "page":       None,
                        "start_char": start,
                        "end_char":   end,
                        "has_image":  True,
                    })
                    image_map[block_index] = _to_png(image_bytes)   # ← normalised

            except Exception as e:
                print(f"[vision] DOCX image failed: {e}")
                continue

    return blocks, image_map


# ── PPTX ──────────────────────────────────────────────────────────────────────

def _parse_pptx(path: Path, image_understanding: bool = False) -> tuple:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs         = Presentation(str(path))
    blocks      = []
    image_map   = {}
    char_cursor = 0

    for slide_num, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:

            # ── text shapes ───────────────────────────────────────────────
            if shape.has_text_frame:
                try:
                    is_title = (
                        shape.is_placeholder
                        and shape.placeholder_format is not None
                        and shape.placeholder_format.idx == 0
                    )
                except Exception:
                    is_title = False

                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if not text:
                        continue

                    elem_type   = "heading" if is_title else "paragraph"
                    start       = char_cursor
                    end         = char_cursor + len(text)
                    char_cursor = end + 1

                    blocks.append({
                        "type":       elem_type,
                        "text":       text,
                        "page":       slide_num,
                        "start_char": start,
                        "end_char":   end,
                        "has_image":  False,
                    })

            # ── image shapes (toggle ON only) ─────────────────────────────
            if image_understanding:
                try:
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        image_bytes = shape.image.blob

                        if len(image_bytes) < 5000:
                            continue

                        print(
                            f"[vision] PPTX slide {slide_num} "
                            f"image '{shape.name}'..."
                        )
                        description = _describe_image_with_llm(
                            image_bytes,
                            context=f"image on slide {slide_num} of presentation"
                        )

                        if description:
                            block_index = len(blocks)
                            start       = char_cursor
                            end         = char_cursor + len(description)
                            char_cursor = end + 1

                            blocks.append({
                                "type":       "paragraph",
                                "text":       f"[Image on slide {slide_num}]: {description}",
                                "page":       slide_num,
                                "start_char": start,
                                "end_char":   end,
                                "has_image":  True,
                            })
                            image_map[block_index] = _to_png(image_bytes)   # ← normalised

                except Exception as e:
                    print(f"[vision] PPTX image failed: {e}")
                    continue

    return blocks, image_map


# ── XLSX ──────────────────────────────────────────────────────────────────────

_XLSX_ROWS_PER_CHUNK = 20


def _parse_xlsx(path: Path) -> list[dict]:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True)
    blocks = []
    char_cursor = 0

    for sheet in wb.worksheets:
        all_rows = []
        for row in sheet.iter_rows(values_only=True):
            cells    = [str(c).strip() if c is not None else "" for c in row]
            row_text = " | ".join(cells).strip(" |")
            if row_text:
                all_rows.append(row_text)

        if not all_rows:
            continue

        header    = all_rows[0]
        data_rows = all_rows[1:]

        full_text = f"[Sheet: {sheet.title}]\n" + "\n".join(all_rows)
        if len(data_rows) <= 50:
            start = char_cursor
            end   = char_cursor + len(full_text)
            char_cursor = end + 1
            blocks.append({
                "type":       "table",
                "text":       full_text,
                "page":       sheet.title,
                "start_char": start,
                "end_char":   end,
            })
            continue

        for i in range(0, len(data_rows), _XLSX_ROWS_PER_CHUNK):
            row_group = data_rows[i: i + _XLSX_ROWS_PER_CHUNK]
            text      = (
                f"[Sheet: {sheet.title}  "
                f"rows {i+1}-{i+len(row_group)}]\n"
                f"{header}\n"
                + "\n".join(row_group)
            )
            start = char_cursor
            end   = char_cursor + len(text)
            char_cursor = end + 1
            blocks.append({
                "type":       "table",
                "text":       text,
                "page":       sheet.title,
                "start_char": start,
                "end_char":   end,
            })

    return blocks


# ── CSV ───────────────────────────────────────────────────────────────────────

_CSV_ROWS_PER_CHUNK = 20


def _parse_csv(path: Path) -> list[dict]:
    import csv

    blocks      = []
    char_cursor = 0
    encodings   = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    raw_rows    = None

    for encoding in encodings:
        try:
            with open(str(path), newline="", encoding=encoding) as f:
                reader   = csv.reader(f)
                raw_rows = list(reader)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if raw_rows is None:
        raise ValueError(
            f"Could not decode '{path.name}'. "
            f"Please save the file as UTF-8 and re-upload."
        )

    all_rows = []
    for row in raw_rows:
        row_text = " | ".join(cell.strip() for cell in row)
        if row_text.strip(" |"):
            all_rows.append(row_text)

    if not all_rows:
        return blocks

    header    = all_rows[0]
    data_rows = all_rows[1:]

    if len(data_rows) <= 50:
        full_text = f"[CSV: {path.name}]\n" + "\n".join(all_rows)
        start     = char_cursor
        end       = char_cursor + len(full_text)
        blocks.append({
            "type":       "table",
            "text":       full_text,
            "page":       path.name,
            "start_char": start,
            "end_char":   end,
        })
        return blocks

    for i in range(0, len(data_rows), _CSV_ROWS_PER_CHUNK):
        row_group = data_rows[i: i + _CSV_ROWS_PER_CHUNK]
        text      = (
            f"[CSV: {path.name}  "
            f"rows {i+1}-{i+len(row_group)}]\n"
            f"{header}\n"
            + "\n".join(row_group)
        )
        start = char_cursor
        end   = char_cursor + len(text)
        char_cursor = end + 1
        blocks.append({
            "type":       "table",
            "text":       text,
            "page":       path.name,
            "start_char": start,
            "end_char":   end,
        })

    return blocks


# ── TXT ───────────────────────────────────────────────────────────────────────

def _parse_txt(path: Path) -> list[dict]:
    blocks      = []
    char_cursor = 0
    encodings   = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    raw         = None

    for encoding in encodings:
        try:
            with open(str(path), encoding=encoding) as f:
                raw = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if raw is None:
        raise ValueError(
            f"Could not decode '{path.name}'. "
            f"Please save the file as UTF-8 and re-upload."
        )

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]

    for para in paragraphs:
        lines     = para.splitlines()
        elem_type = (
            "heading"
            if len(lines) == 1 and len(para) < 80 and not para.endswith(".")
            else "paragraph"
        )

        start = char_cursor
        end   = char_cursor + len(para)
        char_cursor = end + 1

        blocks.append({
            "type":       elem_type,
            "text":       para,
            "page":       None,
            "start_char": start,
            "end_char":   end,
        })

    return blocks


# ── Image files (PNG, JPG, JPEG) ──────────────────────────────────────────────

def _parse_image(path: Path, image_understanding: bool = False) -> tuple:
    """
    Parse a pure image file (PNG, JPG, JPEG).
    RapidOCR extracts any text visible in the image.
    Image is always saved to image_map regardless of toggle.
    GPT-4.1-mini vision describes content if toggle is ON.
    """
    from rapidocr_onnxruntime import RapidOCR

    blocks      = []
    image_map   = {}
    char_cursor = 0

    with open(str(path), "rb") as f:
        image_bytes = f.read()

    # normalise to PNG once upfront
    png_bytes = _to_png(image_bytes)

    # ── RapidOCR for text ────────────────────────────────────────────────
    print(f"[ocr] running OCR on image '{path.name}'...")
    ocr_engine = RapidOCR()
    result, _  = ocr_engine(png_bytes)

    if result:
        result_sorted = sorted(result, key=lambda x: x[0][0][1])
        lines = [
            item[1].strip()
            for item in result_sorted
            if item[1].strip() and item[2] > 0.5
        ]
        if lines:
            ocr_text    = "\n".join(lines)
            block_index = len(blocks)
            start       = char_cursor
            end         = char_cursor + len(ocr_text)
            char_cursor = end + 1

            blocks.append({
                "type":       "paragraph",
                "text":       f"[OCR text]: {ocr_text}",
                "page":       1,
                "start_char": start,
                "end_char":   end,
                "has_image":  True,   # always True — image file IS the content
            })
            image_map[block_index] = png_bytes   # ← always saved
            print(f"[ocr] extracted {len(lines)} lines from image")

    # ── GPT-4.1-mini vision for image understanding (toggle ON only) ──────
    if image_understanding:
        print(f"[vision] describing image '{path.name}'...")
        description = _describe_image_with_llm(
            png_bytes,
            context=f"standalone image file: {path.name}"
        )

        if description:
            block_index = len(blocks)
            start       = char_cursor
            end         = char_cursor + len(description)
            char_cursor = end + 1

            blocks.append({
                "type":       "paragraph",
                "text":       f"[Image description]: {description}",
                "page":       1,
                "start_char": start,
                "end_char":   end,
                "has_image":  True,
            })
            image_map[block_index] = png_bytes   # ← same image referenced again

    if not blocks:
        raise ValueError(
            f"'{path.name}' — no text or content could be extracted."
        )

    return blocks, image_map


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _build_full_text(blocks: list[dict]) -> str:
    return "\n".join(b["text"] for b in blocks)