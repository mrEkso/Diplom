"""Prepare the thesis PDF for the PDF/A conversion in make_pdfa.sh.

Steps:
1. Optionally put a scanned page (the signed Statement of authorship) in place of
   page N, scaled to fit A4 without distortion. Dark strips at the edges of the
   scan are trimmed and the grey paper tone is set to white. The original page object is kept, so links and
   destinations that point to it stay valid.
2. Scale every page that is not A4 (e.g. a scan inserted at pixel size) to fit A4,
   centred.
3. Mark all annotations as printable. PDF/A does not allow non-printing
   annotations, and Ghostscript would otherwise drop all hyperlinks.
4. Rebuild the bookmarks: drop entries that a merge tool named after temporary
   files (hash or *.pdf titles) and add bookmarks for the unnumbered headings
   (Statement of authorship, Abstract, Bibliography).

Usage:
    python prepare.py INPUT OUTPUT [--signature SCAN.pdf] [--signature-page N]
"""

import argparse
import io
import re

from PIL import Image

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, NameObject, NumberObject, RectangleObject
from pypdf.generic import Fit

A4 = (595.276, 841.89)
EXTRA_BOOKMARKS = ["Statement of authorship", "Abstract", "Bibliography"]
JUNK_TITLE = re.compile(r"^[0-9a-f]{16,}|\.pdf$", re.I)


def fit_to_a4(width, height):
    scale = min(A4[0] / width, A4[1] / height)
    tx = (A4[0] - width * scale) / 2
    ty = (A4[1] - height * scale) / 2
    return Transformation().scale(scale).translate(tx, ty)


def clean_scan(page):
    """Return a cleaned copy of a page that holds a single scanned image, or None.

    Trims the dark strips at the edges (table or scanner lid visible next to the
    paper) and lifts the light grey paper tone to white. The aspect ratio is kept.
    """
    xobjects = page.get("/Resources", {}).get("/XObject", {})
    images = [x.get_object() for x in xobjects.values()]
    if len(images) != 1 or images[0].get("/Subtype") != "/Image":
        return None
    im = Image.open(io.BytesIO(images[0]._data)).convert("RGB")
    gray = im.convert("L")
    w, h = gray.size

    def bright(values):
        return sum(values) / len(values) > 240

    def col(x):
        return [gray.getpixel((x, y)) for y in range(0, h, 4)]

    def row(y):
        return [gray.getpixel((x, y)) for x in range(0, w, 4)]

    left, right, top, bottom = 0, w, 0, h
    while left < w // 10 and not bright(col(left)):
        left += 1
    while right > w - w // 10 and not bright(col(right - 1)):
        right -= 1
    while top < h // 10 and not bright(row(top)):
        top += 1
    while bottom > h - h // 10 and not bright(row(bottom - 1)):
        bottom -= 1
    margin = 6  # also drop the soft shadow next to the strip
    box = (left + margin * (left > 0), top + margin * (top > 0),
           right - margin * (right < w), bottom - margin * (bottom < h))
    im = im.crop(box).point(lambda v: min(255, round(v * 255 / 242)))

    buf = io.BytesIO()
    im.save(buf, format="PDF", resolution=300, quality=92)
    print(f"Scan: trimmed {box[0]}/{w - box[2]}/{box[1]}/{h - box[3]} px "
          "(left/right/top/bottom), paper tone set to white")
    return PdfReader(buf).pages[0]


def is_a4(page):
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    return abs(w - A4[0]) < 2 and abs(h - A4[1]) < 2


def flatten_outline(reader, items, level=0, out=None):
    """Return (title, page index, left, top, level), skipping junk entries."""
    out = [] if out is None else out
    last_level = level
    for item in items:
        if isinstance(item, list):
            flatten_outline(reader, item, last_level + 1, out)
            continue
        if JUNK_TITLE.search(str(item.title)):
            last_level = level - 1  # its children move up one level
            continue
        last_level = level
        page = reader.get_destination_page_number(item)
        out.append((str(item.title), page, item.left, item.top, level))
    return out


def first_line(page):
    text = (page.extract_text() or "").strip()
    return text.split("\n", 1)[0].strip() if text else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--signature", help="PDF whose page replaces --signature-page")
    ap.add_argument("--signature-page", type=int, default=3, help="1-based page number")
    ap.add_argument("--signature-source-page", type=int, default=None,
                    help="page in --signature to use (default: same as --signature-page, "
                         "or 1 if the file has a single page)")
    args = ap.parse_args()

    reader = PdfReader(args.input)
    writer = PdfWriter(clone_from=reader)
    entries = flatten_outline(reader, reader.outline)

    if args.signature:
        scan_reader = PdfReader(args.signature)
        src_no = args.signature_source_page or (
            1 if len(scan_reader.pages) == 1 else args.signature_page)
        scan = scan_reader.pages[src_no - 1]
        scan = clean_scan(scan) or scan
        target = writer.pages[args.signature_page - 1]
        target.replace_contents(ContentStream(None, writer))
        target[NameObject("/MediaBox")] = RectangleObject([0, 0, *A4])
        for key in ("/CropBox", "/TrimBox", "/BleedBox", "/ArtBox", "/Rotate"):
            target.pop(key, None)
        target.merge_transformed_page(
            scan, fit_to_a4(float(scan.mediabox.width), float(scan.mediabox.height)))
        print(f"Page {args.signature_page}: replaced with page {src_no} of {args.signature}")

    for no, page in enumerate(writer.pages, 1):
        if not is_a4(page):
            w, h = float(page.mediabox.width), float(page.mediabox.height)
            scan = clean_scan(page)
            if scan is not None:
                page.replace_contents(ContentStream(None, writer))
                sw, sh = float(scan.mediabox.width), float(scan.mediabox.height)
                page.merge_transformed_page(scan, fit_to_a4(sw, sh))
            else:
                page.add_transformation(fit_to_a4(w, h))
            page.mediabox = RectangleObject([0, 0, *A4])
            for key in ("/CropBox", "/TrimBox", "/BleedBox", "/ArtBox"):
                page.pop(key, None)
            print(f"Page {no}: {w:.0f}x{h:.0f} pt scaled to A4")

    n_annots = 0
    for page in writer.pages:
        for annot in page.get("/Annots") or []:
            annot.get_object()[NameObject("/F")] = NumberObject(4)
            n_annots += 1
    print(f"Annotations marked printable: {n_annots}")

    titles = {t for t, *_ in entries}
    for title in EXTRA_BOOKMARKS:
        if title in titles:
            continue
        for no, page in enumerate(reader.pages):
            if first_line(page) == title:
                entries.append((title, no, None, None, 0))
                break
        else:
            if title == "Statement of authorship":  # a scan has no text layer
                entries.append((title, args.signature_page - 1, None, None, 0))
    entries.sort(key=lambda e: (e[1], e[3] is not None, -(e[3] or 0)))

    if "/Outlines" in writer._root_object:
        del writer._root_object["/Outlines"]
    parents = {}
    for title, page, left, top, level in entries:
        fit = Fit.xyz(left, top) if top is not None else Fit.fit()
        parent = parents.get(level - 1) if level > 0 else None
        parents[level] = writer.add_outline_item(title, page, parent=parent, fit=fit)
    print(f"Bookmarks: {len(entries)} ({', '.join(t for t, *_ in entries[:4])}, ...)")

    writer.add_metadata({
        "/Title": "Measuring Factual Knowledge in Text-to-Image Models",
        "/Author": "Davyd Okaianchenko",
        "/Subject": "Bachelor's thesis, TU Dresden",
    })
    writer.page_mode = "/UseOutlines"
    with open(args.output, "wb") as fh:
        writer.write(fh)


if __name__ == "__main__":
    main()
