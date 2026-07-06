#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


PROMPT = """你是一个严谨的中文教材影印页转写员。

请识别图片中所有可见文字，并只输出转写结果，不要解释、不要总结、不要添加图片中不存在的内容。

转写规则：
1. 使用简体中文原文，英文、数字、公式、符号按图片原样保留。
2. 按自然阅读顺序输出。双栏、边注、图表、页脚等要尽量保持它们在书页中的逻辑位置。
3. 正文可以合并扫描换行，保持自然段落；标题单独成行。
4. 表格请尽量转为 Markdown 表格；公式单独成行。
5. 页边注、图注、表注请保留，并用【边注】、【图注】、【表注】标识。
6. 对明显页码、页眉、页脚也保留。
7. 看不清的字用[不清晰]标记，不要猜测。
"""


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def transcribe_page(client: OpenAI, model: str, image_path: Path, max_output_tokens: int) -> str:
    image_base64 = encode_image(image_path)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}",
                    },
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
    )
    return response.output_text.strip()


def combine(page_txt_dir: Path, final_txt: Path, source_pdf: str, model: str):
    with final_txt.open("w", encoding="utf-8") as f:
        f.write("《经济学原理（第7版）宏观经济学分册》LLM OCR 文本\n")
        f.write(f"来源文件：{source_pdf}\n")
        f.write(f"识别模型：{model}\n")
        f.write("说明：由影印版逐页交给视觉大模型转写生成，仍可能存在错字、漏字或版面顺序误差。\n\n")
        for txt in sorted(page_txt_dir.glob("page_*.txt")):
            f.write(txt.read_text(encoding="utf-8").rstrip())
            f.write("\n\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe scanned page images with an OpenAI vision model.")
    parser.add_argument("--images", required=True, help="Directory containing page_001.png style page images.")
    parser.add_argument("--page-text", required=True, help="Directory for per-page transcription files.")
    parser.add_argument("--output", required=True, help="Combined txt output path.")
    parser.add_argument("--source-pdf", required=True, help="Original PDF name recorded in the combined output.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="Vision-capable OpenAI model.")
    parser.add_argument("--max-new", type=int, default=5, help="Maximum new pages to process in this run.")
    parser.add_argument("--start", type=int, default=1, help="First PDF page number to process, 1-based.")
    parser.add_argument("--end", type=int, default=None, help="Last PDF page number to process, 1-based.")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true", help="Re-run pages even when page txt already exists.")
    parser.add_argument("--combine-only", action="store_true", help="Only combine existing per-page txt files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    img_dir = Path(args.images)
    page_txt_dir = Path(args.page_text)
    final_txt = Path(args.output)
    page_txt_dir.mkdir(exist_ok=True)

    images = sorted(img_dir.glob("page_*.png"))
    if not images:
        print(f"No page images found in {img_dir}", file=sys.stderr)
        return 2

    selected = []
    for idx, image in enumerate(images, 1):
        if idx < args.start:
            continue
        if args.end is not None and idx > args.end:
            continue
        selected.append((idx, image))

    if args.combine_only:
        combine(page_txt_dir, final_txt, args.source_pdf, args.model)
        print(f"combined {final_txt}")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Export it first, then rerun this script.", file=sys.stderr)
        return 3

    client = OpenAI()
    processed = 0
    start_time = time.time()

    for idx, image in selected:
        out = page_txt_dir / f"{image.stem}.txt"
        if out.exists() and out.stat().st_size > 0 and not args.overwrite:
            continue

        page_start = time.time()
        text = None
        last_error = None
        for attempt in range(1, 4):
            try:
                text = transcribe_page(client, args.model, image, args.max_output_tokens)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(2**attempt, 10))

        if text is None:
            out.write_text(
                f"===== PDF 第 {idx:03d} 页 =====\n[LLM_OCR_ERROR] {type(last_error).__name__}: {last_error}\n",
                encoding="utf-8",
            )
        else:
            out.write_text(f"===== PDF 第 {idx:03d} 页 =====\n{text}\n", encoding="utf-8")

        processed += 1
        done = len(list(page_txt_dir.glob("page_*.txt")))
        print(
            f"llm_ocr {done}/{len(images)} new={processed} page={idx} last={time.time() - page_start:.1f}s",
            flush=True,
        )

        if processed >= args.max_new:
            break

    page_count = len(list(page_txt_dir.glob("page_*.txt")))
    if page_count == len(images):
        combine(page_txt_dir, final_txt, args.source_pdf, args.model)
        print(f"combined {final_txt}", flush=True)

    print(f"batch_done new={processed} total_pages={page_count} elapsed={(time.time() - start_time) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
