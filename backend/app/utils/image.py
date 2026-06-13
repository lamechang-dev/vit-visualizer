from PIL import Image
from fastapi import UploadFile
import io


async def read_upload_as_image(file: UploadFile) -> Image.Image:
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")
