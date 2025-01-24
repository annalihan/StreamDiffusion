from importlib import import_module
from types import ModuleType
from typing import Dict, Any
from pydantic import BaseModel as PydanticBaseModel, Field
from PIL import Image
import io
import cv2
import numpy as np
from rembg import remove



def get_pipeline_class(pipeline_name: str) -> ModuleType:
    try:
        module = import_module(f"pipelines.{pipeline_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Pipeline {pipeline_name} module not found")

    pipeline_class = getattr(module, "Pipeline", None)

    if pipeline_class is None:
        raise ValueError(f"'Pipeline' class not found in module '{pipeline_name}'.")

    return pipeline_class

def bytes_to_pil_heatmap(image_bytes: bytes) -> Image.Image:
    # 字节数据转成OpenCV图像
    image_np = np.frombuffer(image_bytes, np.uint8 )
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # 高斯模糊
    blurred_image = cv2.GaussianBlur(image, (15,15), 0)

    # 图像转成灰度图，平替生成深度图方法
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # 灰度图转为热力图，热色调
    heatmap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

    _, buffer = cv2.imencode('.png', heatmap)

    # cv2.imshow('heatmap', heatmap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    heatmap_data = buffer.tobytes()

    heatmap_image = Image.open(io.BytesIO(heatmap_data))

    return heatmap_image
    # image = Image.open(io.BytesIO(image_bytes))
    # return image

def bytes_to_pil_canny(image_bytes: bytes) -> Image.Image:
    image_np = np.frombuffer(image_bytes, np.uint8 )
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    edges = cv2.Canny(image, 100, 200)

    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
    _, buffer = cv2.imencode('.png', heatmap)
    # cv2.imshow('heatmap', heatmap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    heatmap_data = buffer.tobytes()

    heatmap_image = Image.open(io.BytesIO(heatmap_data))

    return heatmap_image
    # image = Image.open(io.BytesIO(image_bytes))
    # return image

def bytes_to_pil_rmbg(image_bytes: bytes) -> Image.Image:
    # 字节数据转成OpenCV图像
    image_np = np.frombuffer(image_bytes, np.uint8 )
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    rmbg_image = remove(image)

    _, buffer = cv2.imencode('.jpg', rmbg_image)
    img_data = buffer.tobytes()
    result_image = Image.open(io.BytesIO(img_data))
    return result_image
    # image = Image.open(io.BytesIO(image_bytes))
    # return image


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image

def pil_to_frame(image: Image.Image) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format="JPEG")
    frame_data = frame_data.getvalue()
    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
        + frame_data
        + b"\r\n"
    )


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent
