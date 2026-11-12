"""Picture for the 11-13 weekly report: the lab photo of 11-12 (real PiPER arm, 3D-printed camera mount on the
gripper, a printed mount part, wrist camera view on the monitor), cropped to landscape. No labels (user request).
Input: the phone photo (path as argv[1]); output: weekly_1113.jpg next to this file."""
import sys
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
im = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB")   # 4284 x 5712 (portrait)
W, H = im.size
top = int(280 * W / 675)                                               # crop line read on a 675-wide preview
crop = im.crop((0, top, W, H))
crop.thumbnail((1600, 1600))
crop.save(HERE / "weekly_1113.jpg", quality=88)
print("wrote", HERE / "weekly_1113.jpg", crop.size)
