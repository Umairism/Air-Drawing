"""Generate the Air Drawing application icon."""
from PIL import Image, ImageDraw, ImageFont
import os

SIZE = 256

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# dark rounded-square background
margin = 10
draw.rounded_rectangle(
    [margin, margin, SIZE - margin, SIZE - margin],
    radius=40,
    fill=(30, 30, 40, 255),
)

# stylized hand silhouette (simplified finger shapes)
cx, cy = SIZE // 2, SIZE // 2 + 20

# palm circle
draw.ellipse([cx - 42, cy - 25, cx + 42, cy + 45], fill=(200, 200, 220))

# fingers (5 rounded rects going upward)
finger_data = [
    (-34, -55, 12, 55),   # pinky
    (-17, -80, 12, 65),   # ring
    (0, -90, 12, 70),     # middle
    (17, -75, 12, 60),    # index
    (38, -40, 12, 38),    # thumb (angled a bit)
]
for dx, dy, w, h in finger_data:
    x0 = cx + dx - w
    y0 = cy + dy
    x1 = cx + dx + w
    y1 = cy + dy + h
    draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=(200, 200, 220))

# index fingertip glow (drawing indicator)
glow_x = cx + 17
glow_y = cy - 75
for r, alpha in [(18, 40), (12, 80), (7, 160)]:
    overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.ellipse(
        [glow_x - r, glow_y - r, glow_x + r, glow_y + r],
        fill=(100, 180, 255, alpha),
    )
    img = Image.alpha_composite(img, overlay)
draw = ImageDraw.Draw(img)

# draw a small curvy trail from fingertip
trail_points = [
    (glow_x, glow_y - 10),
    (glow_x + 15, glow_y - 25),
    (glow_x + 5, glow_y - 42),
    (glow_x + 20, glow_y - 55),
]
for i in range(len(trail_points) - 1):
    draw.line([trail_points[i], trail_points[i + 1]], fill=(100, 180, 255, 200), width=3)

# "AD" text at bottom
try:
    font = ImageFont.truetype("arial.ttf", 28)
except OSError:
    font = ImageFont.load_default()
draw.text((cx - 22, cy + 50), "AD", fill=(100, 180, 255), font=font)

# save as .ico with multiple sizes
icon_path = os.path.join(os.path.dirname(__file__), "air_drawing.ico")
img_rgb = img.convert("RGBA")
img_rgb.save(
    icon_path,
    format="ICO",
    sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)
print(f"Icon saved to {icon_path}")
