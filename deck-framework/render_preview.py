#!/usr/bin/env python3
"""macOS-only preview: render each slide of a .pptx to a PNG via Quick Look.

Quick Look only thumbnails the first slide, so for each slide we write a tiny
temp .pptx whose sldIdLst points at that one slide, thumbnail it, and collect
the PNGs under preview/.

Usage:
    python render_preview.py out/Genie_for_Voice_Usecases.pptx [slide_numbers...]
"""
import os
import re
import subprocess
import sys
import zipfile

PPTX = sys.argv[1] if len(sys.argv) > 1 else "out/Genie_for_Voice_Usecases.pptx"
OUTDIR = "preview"
TMPDIR = "preview/_tmp"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

z = zipfile.ZipFile(PPTX)
names = z.namelist()
cache = {n: z.read(n) for n in names}  # read once
pres = cache["ppt/presentation.xml"].decode()
rels = cache["ppt/_rels/presentation.xml.rels"].decode()
slide_rid = {int(m.group(2)): m.group(1) for m in
             re.finditer(r'<Relationship Id="(rId\d+)"[^>]*Target="slides/slide(\d+)\.xml"', rels)}
nslides = len(slide_rid)
want = [int(x) for x in sys.argv[2:]] or list(range(1, nslides + 1))

for s in want:
    one = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>",
                 f'<p:sldIdLst><p:sldId id="256" r:id="{slide_rid[s]}"/></p:sldIdLst>',
                 pres, flags=re.S)
    tmp = os.path.join(TMPDIR, f"s{s}.pptx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as o:
        for n in names:
            o.writestr(n, one if n == "ppt/presentation.xml" else cache[n])
    subprocess.run(["qlmanage", "-t", "-s", "1400", "-o", OUTDIR, tmp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    src = os.path.join(OUTDIR, f"s{s}.pptx.png")
    dst = os.path.join(OUTDIR, f"slide{s:02d}.png")
    if os.path.exists(src):
        os.replace(src, dst)
        print("rendered", dst)
    else:
        print("FAILED", s)
    os.remove(tmp)
print("DONE")
