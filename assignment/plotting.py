from PIL import Image, ImageDraw, ImageFont

def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_loss_curve(losses, path, max_epoch=None, width=1200, height=750):
    all_values = [float(value) for value in losses]
    limit = len(all_values) if max_epoch is None else min(max_epoch, len(all_values))
    values = all_values[:limit]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(24)
    small_font = load_font(18)
    left, top, right, bottom = 110, 80, width - 60, height - 95
    min_loss = min(values)
    max_loss = max(values)
    span = max(max_loss - min_loss, 1e-12)

    draw.rectangle((left, top, right, bottom), outline=(30, 30, 30), width=2)
    for index in range(6):
        ratio = index / 5
        y = bottom - ratio * (bottom - top)
        x = left + ratio * (right - left)
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.line((x, top, x, bottom), fill=(235, 235, 235), width=1)
        loss_value = min_loss + ratio * span
        epoch_value = int(1 + ratio * (limit - 1))
        draw.text((18, y - 10), f"{loss_value:.4f}", fill=(20, 20, 20), font=small_font)
        draw.text((x - 22, bottom + 18), str(epoch_value), fill=(20, 20, 20), font=small_font)

    points = []
    count = len(values)
    for index, value in enumerate(values):
        x = left + index * (right - left) / max(count - 1, 1)
        y = bottom - (value - min_loss) * (bottom - top) / span
        points.append((x, y))

    if len(points) > 1:
        draw.line(points, fill=(25, 95, 180), width=3)

    title = "Training Loss Curve"
    title_box = draw.textbbox((0, 0), title, font=font)
    draw.text(((width - title_box[2]) / 2, 28), title, fill=(20, 20, 20), font=font)
    draw.text(((left + right) / 2 - 35, height - 48), "Epoch", fill=(20, 20, 20), font=font)
    draw.text((18, 38), "Loss", fill=(20, 20, 20), font=font)
    image.save(path)
