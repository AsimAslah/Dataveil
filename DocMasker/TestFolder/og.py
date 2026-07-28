from PIL import Image

INPUT_PATH = "og.png"
OUTPUT_PATH = "og_1200x630.png"
TARGET_SIZE = (1200, 630)
BACKGROUND_COLOR = (0, 0, 0)  # black

# Open image
img = Image.open(INPUT_PATH).convert("RGBA")

# Resize while keeping aspect ratio
img.thumbnail(TARGET_SIZE, Image.LANCZOS)

# Create background canvas
canvas = Image.new("RGBA", TARGET_SIZE, BACKGROUND_COLOR)

# Center the image
x = (TARGET_SIZE[0] - img.width) // 2
y = (TARGET_SIZE[1] - img.height) // 2
canvas.paste(img, (x, y), img)

# Save as PNG
canvas.convert("RGB").save(OUTPUT_PATH, "PNG", optimize=True)

print(f"Saved OG image as {OUTPUT_PATH}")
