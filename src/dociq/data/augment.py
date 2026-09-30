"""Image degradations for the robustness test (PIL only)."""
import io

from PIL import Image, ImageEnhance, ImageFilter


def blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def rotate(img, deg=7):
    return img.rotate(deg, expand=True, fillcolor=(255, 255, 255))


def jpeg(img, quality=20):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def dark(img):
    return ImageEnhance.Brightness(img).enhance(0.5)


DEGRADATIONS = {"blur": blur, "rotate": rotate, "jpeg": jpeg, "dark": dark}


def degrade(img, name: str):
    return DEGRADATIONS[name](img.convert("RGB"))
