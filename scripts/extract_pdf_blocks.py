"""Extract reviewable text blocks from AChemS PDF files.

Usage: python scripts/extract_pdf_blocks.py /path/to/pdfs --output data/extracted_blocks.csv
This is a staging dataset, NOT a validated presentation or abstract dataset.
"""
from pathlib import Path
import argparse
import csv
import re
import fitz

def year_from_name(name):
    match = re.search(r"(?<!\\d)(19\\d{2}|20\\d{2})(?!\\d)", name)
    return int(match.group(1)) if match else None

def extract(pdf_path):
    year = year_from_name(pdf_path.name)
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks", sort=True)
            text_length = sum(len(str(block[4]).strip()) for block in blocks if len(block)>4)
            scanned_or_sparse = text_length < 100
            for block_idx, block in enumerate(blocks, start=1):
                text = re.sub(r"\\s+", " ", str(block[4])).strip()
                if not text:
                    continue
                yield {
                    "year":year or "", "source_pdf":pdf_path.name,
                    "page":page_num, "block":block_idx, "text":text,
                    "needs_ocr":scanned_or_sparse, "review_status":"unreviewed",
                }
            if not blocks:
                yield {"year":year or "", "source_pdf":pdf_path.name,
                       "page":page_num, "block":"", "text":"",
                       "needs_ocr":True, "review_status":"unreviewed"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("pdf_dir",type=Path)
    parser.add_argument("--output",type=Path,default=Path("data/extracted_blocks.csv"))
    args=parser.parse_args()
    pdfs=sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in the provided folder")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fields=["year","source_pdf","page","block","text","needs_ocr","review_status"]
    with args.output.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for pdf in pdfs:
            writer.writerows(extract(pdf))
    print(f"Processed {len(pdfs)} PDFs into {args.output}. Review and OCR are still required.")

if __name__=="__main__":
    main()
