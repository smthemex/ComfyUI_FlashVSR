from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..request import InferenceRequest


def _resize_tensor_video(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Center-crop and resize a [B,C,F,H,W] tensor in its existing range."""
    if height <= 0 or width <= 0:
        raise ValueError("target height and width must be positive")
    source_height, source_width = value.shape[-2:]
    if (source_height, source_width) == (height, width):
        return value
    source_aspect = source_height / source_width
    target_aspect = height / width
    if source_aspect < target_aspect:
        crop_width = max(1, int(source_height / height * width))
        left = max(0, (source_width - crop_width) // 2)
        value = value[..., :, left:left + crop_width]
    else:
        crop_height = max(1, int(source_width / width * height))
        top = max(0, (source_height - crop_height) // 2)
        value = value[..., top:top + crop_height, :]
    batch, channels, frames = value.shape[:3]
    flat = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *value.shape[-2:])
    flat = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
    return flat.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def pil_center_crop_resize(
    frames: np.ndarray,
    target_height: int,
    target_width: int,
    *,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> np.ndarray:
    """Center crop to the requested aspect ratio and resize with PIL.

    This mirrors the geometry used by the original full-video inference path.
    OpenCV is only responsible for decoding; all spatial image operations are
    deliberately performed by PIL.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [F,H,W,3] RGB frames, got {frames.shape}")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target height and width must be positive")

    output = []
    for frame in frames:
        image = Image.fromarray(frame, mode="RGB")
        source_width, source_height = image.size
        if source_width == target_width and source_height == target_height:
            output.append(np.asarray(image))
            continue
        source_aspect = source_height / source_width
        target_aspect = target_height / target_width
        if source_aspect < target_aspect:
            crop_width = int(source_height / target_height * target_width)
            left = (source_width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, source_height))
        else:
            crop_height = int(source_width / target_width * target_height)
            top = (source_height - crop_height) // 2
            image = image.crop((0, top, source_width, top + crop_height))
        image = image.resize((target_width, target_height), resample=resample)
        output.append(np.asarray(image))
    return np.ascontiguousarray(np.stack(output))


def load_lq_video(
    request: InferenceRequest,
    *,
    stream: bool = False,
) -> tuple[torch.Tensor, float]:
    request.validate()
    if request.lq_video is not None:
        value = request.lq_video
        if value.shape[1] != 3 and value.shape[2] == 3:
            value = value.permute(0, 2, 1, 3, 4)
        if value.shape[1] != 3:
            raise ValueError("lq_video channel dimension must be 3")
        if request.height is not None:
            value = _resize_tensor_video(value.float(), request.height, request.width)
        if request.max_frames is not None:
            value = value[:, :, : request.max_frames]
        return value.float(), float(request.fps or 24)

    capture = cv2.VideoCapture(str(Path(request.input_path).expanduser()))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {request.input_path}")
    fps = float(request.fps or capture.get(cv2.CAP_PROP_FPS) or 24)
    frames = []
    try:
        while request.max_frames is None or len(frames) < request.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from: {request.input_path}")
    array = np.stack(frames)
    if request.height is not None or request.width is not None:
        if request.height is None or request.width is None:
            raise ValueError("height and width must be provided together")
        array = pil_center_crop_resize(
            array,
            request.height,
            request.width,
            resample=(Image.Resampling.BICUBIC if stream else Image.Resampling.BILINEAR),
        )
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).float()
    tensor = tensor.div(127.5).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)
    return tensor, fps


def get_lq_video_fps(request: InferenceRequest) -> float:
    request.validate()
    if request.lq_video is not None:
        return float(request.fps or 24)
    capture = cv2.VideoCapture(str(Path(request.input_path).expanduser()))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open input video: {request.input_path}")
        return float(request.fps or capture.get(cv2.CAP_PROP_FPS) or 24)
    finally:
        capture.release()


def iter_lq_video(
    request: InferenceRequest, *, chunk_frames: int, stream: bool = True,
    first_chunk_frames: int | None = None,
):
    """Yield bounded ``[1,3,F,H,W]`` CPU windows from a video file.

    Windows are independent ownership units: callers may release each tensor
    before requesting the next one. The final short window is yielded as-is;
    temporal padding remains the runner's responsibility.
    ``first_chunk_frames`` allows the unique first frame plus one full
    look-ahead region to be read before the first decode segment.
    """
    request.validate()
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if first_chunk_frames is not None and first_chunk_frames <= 0:
        raise ValueError("first_chunk_frames must be positive")
    window_size = first_chunk_frames or chunk_frames
    if request.lq_video is not None:
        value = request.lq_video
        if value.shape[1] != 3 and value.shape[2] == 3:
            value = value.permute(0, 2, 1, 3, 4)
        if value.shape[1] != 3:
            raise ValueError("lq_video channel dimension must be 3")
        value = value[:, :, :request.max_frames] if request.max_frames else value
        if not value.shape[2]:
            raise ValueError("No frames in lq_video")
        start = 0
        while start < value.shape[2]:
            clip = value[:, :, start:start + window_size].float()
            if request.height is not None:
                clip = _resize_tensor_video(clip, request.height, request.width)
            yield clip
            start += window_size
            window_size = chunk_frames
        return
    capture = cv2.VideoCapture(str(Path(request.input_path).expanduser()))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {request.input_path}")
    count = 0
    batch = []
    try:
        while request.max_frames is None or count < request.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(batch) == window_size:
                value = _frames_to_tensor(batch, request, stream=stream)
                batch.clear()
                yield value
                del value
                window_size = chunk_frames
        if batch:
            yield _frames_to_tensor(batch, request, stream=stream)
        if not count:
            raise ValueError(f"No frames decoded from: {request.input_path}")
    finally:
        capture.release()


def _frames_to_tensor(frames, request, *, stream):
    array = np.stack(frames)
    if request.height is not None:
        if request.width is None:
            raise ValueError("height and width must be provided together")
        array = pil_center_crop_resize(
            array, request.height, request.width,
            resample=Image.Resampling.BICUBIC if stream else Image.Resampling.BILINEAR,
        )
    return torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).permute(1, 0, 2, 3).unsqueeze(0)
