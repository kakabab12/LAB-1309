"""Put the rendered poster in the pptx file thumbnail (the template's thumbnail shows its sample poster).
usage: python3 paper/poster/set_thumbnail.py POSTER.pptx PREVIEW.png"""
import io
import shutil
import sys
import tempfile
import zipfile

from PIL import Image

pptx, png = sys.argv[1:3]
im = Image.open(png).convert("RGB")
im.thumbnail((192, 256), Image.LANCZOS)
buf = io.BytesIO()
im.save(buf, "JPEG", quality=85)
with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
    with zipfile.ZipFile(pptx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = buf.getvalue() if item.filename == "docProps/thumbnail.jpeg" else zin.read(item.filename)
            zout.writestr(item, data)
shutil.move(tmp.name, pptx)
print("thumbnail", im.size, "->", pptx)
