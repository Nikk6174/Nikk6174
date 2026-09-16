#!/usr/bin/env python3
"""Turn a photo into ascii.svg using braille characters for high-resolution output.

Braille characters (U+2800-U+28FF) encode a 2x4 dot grid per character,
giving 8x the effective resolution of standard ASCII art. At 90 columns,
this produces the equivalent of ~180x320 pixel detail.

    pip install pillow numpy opencv-python-headless rembg onnxruntime
    python3 scripts/make_portrait.py assets/photo.png

The first run downloads a ~176 MB background-removal model, once.
"""
import argparse
import sys

import cv2
import numpy as np
from PIL import Image
from rembg import remove

COLS = 90                  # character columns in the output
CLAHE_CLIP = 3.0           # higher amplifies skin texture into noise
CURVE = 1.7                # the darkening curve -- the difference-maker
CROP_BOTTOM = 0.0          # fraction to trim off the bottom (torso, chair)
ROW_RATIO = 0.48           # monospace cells are about twice as tall as wide

# Braille dot mapping: each braille char is a 2x4 grid of dots
# Unicode braille starts at U+2800; each dot maps to a specific bit:
#   Position (row,col) -> bit value:
#   (0,0)->0x01  (0,1)->0x08
#   (1,0)->0x02  (1,1)->0x10
#   (2,0)->0x04  (2,1)->0x20
#   (3,0)->0x40  (3,1)->0x80
BRAILLE_BASE = 0x2800
DOT_MAP = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]

FG_LIGHT = "#6e7681"       # readable on GitHub light
FG_DARK = "#c9d1d9"        # dark-mode variant
CHAR_W = 7.74              # 0.600 em at FONT_SIZE
FONT_SIZE = 12.9
LINE_H = 15
ROW_DELAY = 0.09           # per-row stagger, seconds
FAMILY = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def prep(path, crop=None):
    """Cut out the background, even the local contrast, then darken."""
    src = Image.open(path).convert("RGBA")
    if crop:
        src = src.crop(crop)

    cut = remove(src)
    alpha = np.array(cut.split()[-1])

    # Composite onto white so everything outside the subject maps to blank
    white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    gray = np.array(Image.alpha_composite(white, cut).convert("L"))

    gray = cv2.bilateralFilter(gray, 11, 50, 50)      # smooth skin, keep edges
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                           tileGridSize=(8, 8)).apply(gray)
    gray = (255.0 * (gray / 255.0) ** CURVE).astype("uint8")
    gray[alpha < 20] = 255                              # force the matte to white
    return Image.fromarray(gray)


def floyd_steinberg_dither(img_array):
    """Apply Floyd-Steinberg dithering to a grayscale image.

    Returns a boolean array where True = dark (dot ON).
    This produces far more detail than simple thresholding, preserving
    gradients and subtle features like glasses frames and facial contours.
    """
    h, w = img_array.shape
    buf = img_array.astype(np.float64)

    for y in range(h):
        for x in range(w):
            old = buf[y, x]
            new = 0.0 if old < 128 else 255.0
            buf[y, x] = new
            err = old - new
            if x + 1 < w:
                buf[y, x + 1] += err * 7.0 / 16.0
            if y + 1 < h:
                if x - 1 >= 0:
                    buf[y + 1, x - 1] += err * 3.0 / 16.0
                buf[y + 1, x] += err * 5.0 / 16.0
                if x + 1 < w:
                    buf[y + 1, x + 1] += err * 1.0 / 16.0

    return buf < 128  # True = dark dot


def to_braille_lines(img, cols=COLS):
    """Convert a grayscale image to braille character lines.

    Each braille character covers a 2x4 pixel area, giving 8x the
    effective resolution of a single ASCII character.
    """
    w, h = img.size
    if CROP_BOTTOM:
        img = img.crop((0, 0, w, int(h * (1 - CROP_BOTTOM))))
        w, h = img.size

    # Calculate character grid dimensions
    char_rows = int(cols * (h / w) * ROW_RATIO)

    # Pixel grid: 2 pixels per column, 4 pixels per row
    pixel_w = cols * 2
    pixel_h = char_rows * 4

    img = img.resize((pixel_w, pixel_h), Image.LANCZOS)
    pixels = np.array(img)

    # Dither for maximum detail
    dark = floyd_steinberg_dither(pixels)

    lines = []
    for row in range(char_rows):
        line = []
        for col in range(cols):
            code = 0
            for dy in range(4):
                for dx in range(2):
                    py = row * 4 + dy
                    px = col * 2 + dx
                    if py < dark.shape[0] and px < dark.shape[1] and dark[py, px]:
                        code |= DOT_MAP[dy][dx]
            line.append(chr(BRAILLE_BASE + code))
        lines.append("".join(line).rstrip())

    # Trim empty lines from top and bottom
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return lines


def build_svg(lines, cols=COLS):
    pad = 14
    width = int(cols * CHAR_W + pad * 2)
    height = len(lines) * LINE_H + pad * 2

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}" '
         f'font-family="{FAMILY}">',
         f'<style>.a{{fill:{FG_LIGHT}}}'
         f'@media(prefers-color-scheme:dark){{.a{{fill:{FG_DARK}}}}}</style>']

    for i, line in enumerate(lines):
        y = pad + i * LINE_H
        begin = f"{i * ROW_DELAY:.2f}s"
        end = f"{(i + 1) * ROW_DELAY:.2f}s"
        w = max(len(line), 1) * CHAR_W
        safe = (line.replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;"))

        p.append(f'<clipPath id="c{i}"><rect x="{pad}" y="{y}" '
                 f'height="{LINE_H}" width="0">'
                 f'<animate attributeName="width" from="0" to="{w:.1f}" '
                 f'begin="{begin}" dur="{ROW_DELAY}s" fill="freeze"/>'
                 f'</rect></clipPath>')
        p.append(f'<g clip-path="url(#c{i})"><text xml:space="preserve" '
                 f'x="{pad}" y="{y + 11.2:.1f}" class="a" '
                 f'font-size="{FONT_SIZE}">{safe}</text></g>')
        # the cursor: a small block riding the wipe edge, gone once the row lands
        p.append(f'<rect y="{y + 1}" width="6" height="12" class="a" '
                 f'opacity="0">'
                 f'<animate attributeName="x" from="{pad}" to="{pad + w:.1f}" '
                 f'begin="{begin}" dur="{ROW_DELAY}s" fill="freeze"/>'
                 f'<set attributeName="opacity" to="0.8" begin="{begin}"/>'
                 f'<set attributeName="opacity" to="0" begin="{end}"/></rect>')

    p.append("</svg>")
    return "".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photo")
    ap.add_argument("out", nargs="?", default="ascii.svg")
    ap.add_argument("--crop", help="left,top,right,bottom, applied first -- crop "
                                   "tight to the head so the whole grid goes to "
                                   "the face")
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--preview", action="store_true",
                    help="print the braille art to the terminal as well")
    args = ap.parse_args()

    crop = None
    if args.crop:
        parts = [int(v) for v in args.crop.split(",")]
        if len(parts) != 4:
            sys.exit("--crop needs four numbers: left,top,right,bottom")
        crop = tuple(parts)

    lines = to_braille_lines(prep(args.photo, crop), cols=args.cols)
    if args.preview:
        print("\n".join(lines))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_svg(lines, cols=args.cols))
    print(f"wrote {args.out} -- {len(lines)} rows, {args.cols} columns (braille)")


if __name__ == "__main__":
    main()
