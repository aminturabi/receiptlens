"""Field-level evaluation: predicted JSON vs ground truth.

Metrics (per field group): precision / recall / F1.
  - items.name : fuzzy match (difflib ratio >= 0.8) between GT and predicted names
  - items.price: exact numeric match, only on name-matched pairs
  - subtotal / tax / total: exact numeric match
Also reports mean latency per receipt.
"""
import difflib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from receiptlens import ReceiptExtractor  # noqa: E402


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def norm_num(s: str) -> str:
    s = (s or "").replace(",", "").strip()
    m = re.search(r"\d+\.\d+|\d+", s)
    if not m:
        return ""
    return f"{float(m.group()):.2f}"


def match_names(gt_items, pred_items, thresh: float = 0.8):
    """Greedy 1:1 matching of GT item names to predicted names."""
    pairs = []
    used = set()
    for g in gt_items:
        gn = norm_text(g.get("name", ""))
        best, best_i = 0.0, -1
        for i, p in enumerate(pred_items):
            if i in used:
                continue
            r = difflib.SequenceMatcher(None, gn, norm_text(p.get("name", ""))).ratio()
            if r > best:
                best, best_i = r, i
        if best >= thresh:
            used.add(best_i)
            pairs.append((g, pred_items[best_i]))
    return pairs


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(data_dir: str, limit: int | None = None) -> dict:
    data = Path(data_dir)
    gt_rows = [json.loads(l) for l in open(data / "gt.jsonl")]
    if limit:
        gt_rows = gt_rows[:limit]

    ex = ReceiptExtractor()
    agg = {"name_tp": 0, "name_fp": 0, "name_fn": 0,
           "price_ok": 0, "price_n": 0,
           "sub_ok": 0, "tax_ok": 0, "tot_ok": 0, "n": 0,
           "lat": []}
    per_receipt = []

    for row in gt_rows:
        img_path = data / row["image_file"]
        t0 = time.time()
        res = ex.extract_bytes(img_path.read_bytes())
        dt = time.time() - t0
        pred_items = [i.model_dump() for i in res.items]

        pairs = match_names(row["items"], pred_items)
        name_tp = len(pairs)
        name_fp = len(pred_items) - name_tp
        name_fn = len(row["items"]) - name_tp
        price_ok = sum(1 for g, p in pairs if norm_num(g["price"]) == norm_num(p.get("price", "")))
        sub_ok = norm_num(row["subtotal"]) == norm_num(res.subtotal or "")
        tax_ok = norm_num(row["tax"]) == norm_num(res.tax or "")
        tot_ok = norm_num(row["total"]) == norm_num(res.total or "")

        agg["name_tp"] += name_tp; agg["name_fp"] += name_fp; agg["name_fn"] += name_fn
        agg["price_ok"] += price_ok; agg["price_n"] += name_tp
        agg["sub_ok"] += sub_ok; agg["tax_ok"] += tax_ok; agg["tot_ok"] += tot_ok
        agg["n"] += 1; agg["lat"].append(dt)
        per_receipt.append({"file": row["image_file"], "name_tp": name_tp,
                            "name_fn": name_fn, "price_ok": price_ok,
                            "sub_ok": sub_ok, "tax_ok": tax_ok, "tot_ok": tot_ok,
                            "latency_s": round(dt, 1)})

    p, r, f = prf(agg["name_tp"], agg["name_fp"], agg["name_fn"])
    report = {
        "n_receipts": agg["n"],
        "item_name": {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)},
        "item_price_given_match": round(agg["price_ok"] / agg["price_n"], 3) if agg["price_n"] else 0.0,
        "subtotal_acc": round(agg["sub_ok"] / agg["n"], 3),
        "tax_acc": round(agg["tax_ok"] / agg["n"], 3),
        "total_acc": round(agg["tot_ok"] / agg["n"], 3),
        "mean_latency_s": round(sum(agg["lat"]) / len(agg["lat"]), 1),
        "per_receipt": per_receipt,
    }
    return report


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "eval/data/synth"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    report = evaluate(data_dir, limit)
    out = Path(data_dir) / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    r = report
    print(f"\nReceiptLens eval — {r['n_receipts']} receipts")
    print(f"  item name   P={r['item_name']['precision']} R={r['item_name']['recall']} F1={r['item_name']['f1']}")
    print(f"  item price  acc (on matched names) = {r['item_price_given_match']}")
    print(f"  subtotal acc = {r['subtotal_acc']}   tax acc = {r['tax_acc']}   total acc = {r['total_acc']}")
    print(f"  mean latency = {r['mean_latency_s']}s / receipt (CPU)")
    print(f"  report -> {out}")


if __name__ == "__main__":
    main()
