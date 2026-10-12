"""Generate a synthetic receipt eval set with exact ground truth.

Each receipt is a thermal-paper-style image; GT is stored in the same
normalized schema the pipeline outputs, so scoring is exact.
"""
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(20261012)

STORES = [
    ("FRESH MART", "123 Market Street", "555-0134"),
    ("CITY BAKERY", "45 Baker Lane", "555-7788"),
    ("GREEN GROCER", "9 Orchard Road", "555-2211"),
    ("DAILY NEEDS", "77 Station Ave", "555-9900"),
    ("SUNRISE FOODS", "210 River Blvd", "555-3344"),
]

PRODUCTS = [
    ("Milk 1L", 3.50), ("Eggs 12pk", 4.20), ("Bread Loaf", 2.75),
    ("Apples 1kg", 5.10), ("Chicken Breast 500g", 7.99), ("Rice 2kg", 6.40),
    ("Pasta 500g", 1.99), ("Tomato Sauce", 2.49), ("Cheese 250g", 4.99),
    ("Yogurt 4pk", 3.79), ("Bananas 1kg", 2.20), ("Coffee 250g", 8.50),
    ("Tea Bags 50ct", 3.30), ("Sugar 1kg", 2.10), ("Salt 500g", 1.20),
    ("Olive Oil 500ml", 9.99), ("Butter 250g", 3.60), ("Juice 1L", 2.99),
    ("Cookies 300g", 2.85), ("Chips 150g", 1.75),
]

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def load_fonts(scale: int = 2):
    try:
        return (ImageFont.truetype(FONT_B, int(26 * scale)),
                ImageFont.truetype(FONT_M, int(22 * scale)),
                ImageFont.truetype(FONT_M, int(19 * scale)))
    except OSError:
        d = ImageFont.load_default()
        return d, d, d


def render_receipt(store, items, tax_rate, out_path, messy: bool):
    W = 560 * 2
    fb, fm, fsmall = load_fonts()
    lines = []
    # small left-aligned header (like real receipts) — keeps the model from
    # reading the store name as a menu item
    lines.append(("H", store[0], True))
    lines.append(("H", store[1] + "  Tel: " + store[2], False))
    lines.append(("L", "-" * 42, False))
    for name, price in items:
        lines.append(("I", (name, f"{price:.2f}"), False))
    lines.append(("L", "-" * 34, False))
    subtotal = sum(p for _, p in items)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    lines.append(("R", ("Subtotal", f"{subtotal:.2f}"), False))
    lines.append(("R", (f"Tax ({tax_rate*100:.0f}%)", f"{tax:.2f}"), False))
    lines.append(("R", ("TOTAL", f"{total:.2f}"), True))
    lines.append(("L", "-" * 42, False))
    lines.append(("H", "Thank you, visit again!", False))

    H = 60 + len(lines) * 44 + 40
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 44
    for kind, payload, bold in lines:
        font = fb if bold else fm
        if kind == "H":
            font = fb if bold else fsmall
            d.text((72, y), payload, fill="black", font=font)
        elif kind == "I":
            name, price = payload
            d.text((72, y), name, fill="black", font=font)
            d.text((W - 72, y), price, fill="black", font=font, anchor="ra")
        elif kind == "R":
            label, val = payload
            d.text((72, y), label, fill="black", font=font)
            d.text((W - 72, y), val, fill="black", font=font, anchor="ra")
        else:
            d.text((72, y), payload, fill="black", font=font)
        y += 44

    if messy:
        # slight rotation + brightness jitter + light blur = "messy scan"
        img = img.rotate(random.uniform(-1.8, 1.8), expand=True, fillcolor="white")
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    img.save(out_path)
    return {
        "merchant": store[0],
        "items": [{"name": n, "price": f"{p:.2f}"} for n, p in items],
        "subtotal": f"{subtotal:.2f}",
        "tax": f"{tax:.2f}",
        "total": f"{total:.2f}",
    }


def main(n: int = 24, outdir: str = "eval/data/synth"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    gt_path = out / "gt.jsonl"
    with open(gt_path, "w") as f:
        for i in range(n):
            store = random.choice(STORES)
            k = random.randint(3, 7)
            items = random.sample(PRODUCTS, k)
            tax_rate = random.choice([0.0, 0.05, 0.08, 0.10])
            messy = i % 3 == 2  # every third receipt is a "messy scan"
            fname = f"receipt_{i:02d}{'_messy' if messy else ''}.png"
            gt = render_receipt(store, items, tax_rate, str(out / fname), messy)
            gt["image_file"] = fname
            f.write(json.dumps(gt) + "\n")
    print(f"wrote {n} receipts -> {out}")


if __name__ == "__main__":
    main()
