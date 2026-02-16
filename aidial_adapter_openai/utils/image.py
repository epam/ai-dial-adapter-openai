import io

from PIL import Image, ImageOps

from aidial_adapter_openai.utils.resource.base import Resource

_CONTENT_TYPE_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

_FORMAT_TO_CONTENT_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def crop_image_file(
    *, resource: Resource, width: int, height: int
) -> Resource | None:
    if width <= 0 or height <= 0:
        return None

    content_type = resource.type.lower()
    if not content_type.startswith("image/"):
        return None

    try:
        with Image.open(io.BytesIO(resource.data)) as img:
            out_format = (
                _CONTENT_TYPE_TO_FORMAT.get(content_type) or img.format or "PNG"
            )

            # JPEG cannot store RGBA
            if out_format.upper() == "JPEG" and img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            fitted = ImageOps.fit(
                img,
                (width, height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )

            buf = io.BytesIO()

            save_kwargs: dict = {}
            if out_format.upper() == "JPEG":
                save_kwargs.update({"quality": 95, "optimize": True})
            elif out_format.upper() == "PNG":
                save_kwargs.update({"optimize": True})

            fitted.save(buf, format=out_format, **save_kwargs)

            new_content_type = _FORMAT_TO_CONTENT_TYPE.get(
                out_format.upper(), content_type
            )

            return Resource(type=new_content_type, data=buf.getvalue())

    except Exception:
        return None
