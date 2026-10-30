#!/usr/bin/env python3
"""Verify the Car License Project Word Lab Manual against the AI3 checklist.

Parses both generated .docx files with python-docx, asserts every mandatory
checklist item, and reports structure + hashes.  Exit code 0 = all pass.

Usage: .venv/bin/python verify_manual.py [path_to_local_docx]
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

from docx import Document

LOCAL = Path("/Users/koil/Lab_Manual_Car_Sign.docx")
ONEDRIVE = Path(
    "/Users/koil/Library/CloudStorage/OneDrive-HKUSTConnect/"
    "Student_Helper/TF Lite Training Manual - Car Sign.docx"
)

RESULTS = []  # (check_id, passed: bool, detail: str)


def check(check_id: str, ok: bool, detail: str) -> None:
    RESULTS.append((check_id, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {check_id}: {detail}")


def norm(s: str) -> str:
    return " ".join(s.split())


def collect_text(doc) -> tuple[str, list[str], str]:
    """All text (paragraphs + table cells), heading list, body paragraphs."""
    chunks: list[str] = []
    headings: list[str] = []
    body_paras: list[str] = []
    for p in doc.paragraphs:
        chunks.append(p.text)
        body_paras.append(p.text)
        if p.style is not None and p.style.name.startswith("Heading"):
            headings.append(norm(p.text))
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                chunks.append(cell.text)
    return norm("\n".join(chunks)), headings, norm("\n".join(body_paras))


def file_stats(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "size": len(data),
        "md5": hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> int:
    for p in (LOCAL, ONEDRIVE):
        if not p.exists():
            check("F0-files", False, f"missing: {p}")
            return 1

    st_local = file_stats(LOCAL)
    st_one = file_stats(ONEDRIVE)
    print("\n== File stats ==")
    for st in (st_local, st_one):
        print(f"  {st['path']}\n    size={st['size']} bytes  md5={st['md5']}  sha256={st['sha256']}  mtime={st['mtime']}")

    check(
        "F1-byte-identical",
        st_local["sha256"] == st_one["sha256"],
        "both destinations byte-identical (sha256 match)",
    )
    check("F2-size", st_local["size"] > 1_000_000, f"size {st_local['size']} bytes (embedded figures expected > 1 MB)")

    doc = Document(str(LOCAL))
    all_text, headings, body_text = collect_text(doc)

    print("\n== Structure ==")
    print(f"  paragraphs: {len(doc.paragraphs)}  tables: {len(doc.tables)}  "
          f"inline_shapes: {len(doc.inline_shapes)}  headings: {len(headings)}")
    check("S1-paragraphs", len(doc.paragraphs) >= 60, f"{len(doc.paragraphs)} body paragraphs (>= 60)")
    check("S2-tables", len(doc.tables) >= 11, f"{len(doc.tables)} tables (>= 11: callouts, boxes, data tables)")
    check("S3-figures", len(doc.inline_shapes) == 11, f"exactly 11 figures, found {len(doc.inline_shapes)}")
    check("S4-sections", all(h in " | ".join(headings) for h in (
        "1. Opening the Project", "2. Sample Area — Part A", "3. Sample Area — Part B",
        "4. Train Area", "5. Preview Area", "6. Car & Checkpoint Physical Placement Specification",
        "7. On-Car Verification")), "all 7 numbered section headings present")

    print("\n== Checklist 1: Top Navigation & Multi-Plate Gatekeeper (License List) ==")
    check("C1.1-navLicenseList", "#navLicenseList" in all_text,
          "manual names the top-bar button id #navLicenseList")
    check("C1.2-next-to-settings", ("🪪 License List" in all_text and "⚙ Settings" in all_text),
          "License List described as next to ⚙ Settings")
    check("C1.3-keyboard-tip", ("Enter to save" in all_text and "Esc to exit Settings" in all_text),
          "keyboard tip present: Enter = Save Settings, Esc = Exit Settings")
    check("C1.4-sign_sequences", "sign_sequences" in all_text and '["A12", "B18"]' in all_text,
          "sign_sequences setting explained with [\"A12\", \"B18\"] example")
    check("C1.5-union", "['A', '1', '2', 'B', '8']" in all_text,
          "unique-character union example [\"A12\",\"B18\"] -> ['A','1','2','B','8']")
    check("C1.6-emnist", ("EMNIST" in all_text and "24" in all_text and "28×28" in all_text),
          "EMNIST 24-sample auto-fill described in the model's 28×28 glyph format")
    check("C1.7-background-default",
          "Use _background noise class to replace OOD rejection" in all_text and "False (Off)" in all_text,
          "advanced setting '_background noise class to replace OOD rejection' with default False (Off)")

    print("\n== Checklist 2: Physical Placement Specification & 4 Reserved Placeholders ==")
    placement_heading = next((h for h in headings if "Car & Checkpoint Physical Placement Specification" in h), None)
    check("C2.1-placement-section",
          placement_heading is not None and "小车与检查站物理摆放规范" in placement_heading,
          f"dedicated section heading found: {placement_heading!r}")
    reserved_labels = [f"Reserved Image {i}" for i in (1, 2, 3, 4)]
    reserved_hits = {lbl: (lbl in all_text) for lbl in reserved_labels}
    box_paras = [
        t.cell(0, 0).paragraphs for t in doc.tables
        if len(t.rows) == 1 and len(t.columns) == 1
        and t.cell(0, 0).paragraphs
        and t.cell(0, 0).paragraphs[0].text.strip().startswith("📷 [Reserved Image")
    ]
    check("C2.2-four-labels", all(reserved_hits.values()),
          f"all 4 reserved image labels present: {reserved_hits}")
    check("C2.3-four-boxes", len(box_paras) == 4,
          f"4 reserved placeholder boxes (single-cell frame tables), found {len(box_paras)}")
    captions_ok = all(
        len(ps) >= 2 and "(Reserved space — drop the photo here)" in ps[1].text
        for ps in box_paras
    )
    check("C2.4-box-captions", captions_ok,
          "each placeholder box carries the '(Reserved space — drop the photo here)' caption")
    check("C2.5-rfid-car-checkpoint", ("RFID Reader" in all_text and "45°" in all_text),
          "placement text covers RFID reader alignment, map edge and 45° checkpoint base")

    print("\n== Checklist 3: Preprocessing, 3-Slot Preview & Firmware Deployment ==")
    check("C3.1-preprocess-modal", "#classPreprocessModal" in all_text,
          "Edit page named as #classPreprocessModal")
    check("C3.2-auto-lock", "In Auto (find sign) mode the app scans the photo and locks a green ROI" in all_text,
          "Auto mode described as locking the ROI onto the character")
    check("C3.3-lum-thr", "Lum Thr" in all_text,
          "Lum Thr adjustment described")
    check("C3.4-save-sample", "Save Sample (#classPreprocessSave)" in all_text,
          "thresholds written to the sample via Save Sample (#classPreprocessSave)")
    check("C3.5-slot-preview", ("WANT" in all_text and "confidence" in all_text),
          "3-slot Preview panel: WANT vs predicted character and confidence %")
    check("C3.6-uart-params", ("921600" in all_text and "8N1" in all_text),
          "UART 921600 baud, 8N1 described")
    check("C3.7-uart-pins", ("GPIO10" in all_text and "GPIO11" in all_text
                             and "GPIO44" in all_text and "GPIO43" in all_text),
          "UART pins: P4 RX=GPIO10/TX=GPIO11, S3 RX=GPIO44/TX=GPIO43")
    check("C3.8-frame-layout",
          "0xAA, 0x55, 0x01, frame_lo, frame_hi, label_id, confidence, flags, xor_checksum" in all_text.replace("[", "").replace("]", ""),
          "9-byte frame layout [0xAA, 0x55, 0x01, frame_lo, frame_hi, label_id, confidence, flags, xor_checksum]")
    check("C3.9-match", ("0x02" in all_text and "kTargetSignCount-1" in all_text),
          "MATCH flags=0x02 -> label_id = plate index (0 … kTargetSignCount-1)")
    check("C3.10-mismatch", ("0xE2" in all_text and "0xFE" in all_text),
          "MISMATCH flags=0xE2 -> label_id = 0xFE")
    check("C3.11-no-sign", ("0xF2" in all_text and "0xFF" in all_text),
          "NO SIGN flags=0xF2 -> label_id = 0xFF")
    check("C3.12-verdict-names", all(v in all_text for v in ("MATCH", "MISMATCH", "NO SIGN")),
          "MATCH / MISMATCH / NO SIGN verdict names explained")
    check("C3.13-serial-line",
          "SIGN: MATCH (YES) plate=A12" in all_text,
          "Serial Monitor example line present")

    print("\n== Summary ==")
    failed = [r for r in RESULTS if not r[1]]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("  FAILED:")
        for cid, _, det in failed:
            print(f"    - {cid}: {det}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
