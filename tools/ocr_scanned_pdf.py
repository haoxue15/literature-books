#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image
from rapidocr_onnxruntime import RapidOCR


def box_bounds(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def ordered_lines(result, width):
    rows = []
    for item in result or []:
        box, text, score = item[0], str(item[1]).strip(), float(item[2])
        if not text or score < 0.45:
            continue
        x0, y0, x1, y1 = box_bounds(box)
        rows.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "cx": (x0 + x1) / 2,
                "text": text,
            }
        )

    if not rows:
        return []

    centers = sorted(r["cx"] for r in rows)
    split = None
    best_gap = 0
    for a, b in zip(centers, centers[1:]):
        gap = b - a
        mid = (a + b) / 2
        left_count = sum(c <= mid for c in centers)
        right_count = len(centers) - left_count
        if gap > best_gap and gap > width * 0.16 and left_count >= 3 and right_count >= 3:
            best_gap = gap
            split = mid

    if split is None:
        return [r["text"] for r in sorted(rows, key=lambda r: (r["y0"], r["x0"]))]

    left = [r for r in rows if r["cx"] <= split]
    right = [r for r in rows if r["cx"] > split]
    lines = [r["text"] for r in sorted(left, key=lambda r: (r["y0"], r["x0"]))]
    if right:
        lines.append("")
        lines.extend(r["text"] for r in sorted(right, key=lambda r: (r["y0"], r["x0"])))
    return lines


def combine(page_txt_dir: Path, final_txt: Path, source_pdf: str):
    with final_txt.open("w", encoding="utf-8") as f:
        f.write("《经济学原理（第7版）宏观经济学分册》OCR 文本\n")
        f.write(f"来源文件：{source_pdf}\n")
        f.write("说明：由影印版逐页 OCR 生成，可能存在错字、漏字、页边注混入正文等问题。\n\n")
        for txt in sorted(page_txt_dir.glob("page_*.txt")):
            f.write(txt.read_text(encoding="utf-8").rstrip())
            f.write("\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--page-text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-pdf", required=True)
    parser.add_argument("--max-new", type=int, default=10)
    args = parser.parse_args()

    img_dir = Path(args.images)
    page_txt_dir = Path(args.page_text)
    final_txt = Path(args.output)
    page_txt_dir.mkdir(exist_ok=True)

    images = sorted(img_dir.glob("page_*.png"))
    ocr = RapidOCR()
    processed = 0
    start = time.time()

    for idx, img in enumerate(images, 1):
        out = page_txt_dir / f"{img.stem}.txt"
        if out.exists() and out.stat().st_size > 0:
            continue

        page_start = time.time()
        try:
            with Image.open(img) as im:
                width = im.width
            result, _ = ocr(str(img), use_cls=False)
            lines = ordered_lines(result, width)
            text = f"===== PDF 第 {idx:03d} 页 =====\n" + "\n".join(lines).strip() + "\n"
            out.write_text(text, encoding="utf-8")
        except Exception as exc:
            out.write_text(
                f"===== PDF 第 {idx:03d} 页 =====\n[OCR_ERROR] {type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )

        processed += 1
        done = len(list(page_txt_dir.glob("page_*.txt")))
        print(
            f"ocr {done}/{len(images)} new={processed} page={idx} last={time.time() - page_start:.1f}s",
            flush=True,
        )
        if processed >= args.max_new:
            break

    page_count = len(list(page_txt_dir.glob("page_*.txt")))
    if page_count == len(images):
        combine(page_txt_dir, final_txt, args.source_pdf)
        print(f"combined {final_txt}", flush=True)

    print(f"batch_done new={processed} total_pages={page_count} elapsed={(time.time() - start) / 60:.1f}m")


if __name__ == "__main__":
    main()
