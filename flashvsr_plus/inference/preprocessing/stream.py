from __future__ import annotations

from collections.abc import Iterator

import torch

from .padding import target_frame_count


class TemporalVideoStream:
    """Expand a source timeline on demand and release consumed LQ frames.

    Only EOF introduces temporal padding. Frame indices refer to the expanded
    model timeline, including its unique first frame.
    """

    def __init__(self, chunks: Iterator[torch.Tensor], compression: int, model_compression: int = 4):
        if compression not in (1, 2, 4) or model_compression % compression:
            raise ValueError("invalid temporal compression")
        self.chunks = iter(chunks)
        self.compression = compression
        self.repeat_factor = model_compression // compression
        self.source_frames = 0
        self.effective_frames = 0
        self.expanded_frames = 0
        self.eof = False
        self.buffer = None
        self.offset = 0
        self.last_frame = None
        self.max_buffer_frames = 0

    def _append(self, value: torch.Tensor) -> None:
        if not value.shape[2]:
            return
        self.buffer = value.clone() if self.buffer is None else torch.cat((self.buffer, value), dim=2)
        self.expanded_frames += value.shape[2]
        self.max_buffer_frames = max(self.max_buffer_frames, self.buffer.shape[2])

    def ensure(self, stop: int) -> bool:
        while self.expanded_frames < stop and not self.eof:
            try:
                value = next(self.chunks)
            except StopIteration:
                self.eof = True
                if not self.source_frames:
                    raise ValueError("No frames decoded from input video")
                self.effective_frames = target_frame_count(self.source_frames, self.compression)
                missing = (self.effective_frames - self.source_frames) * self.repeat_factor
                self._append(self.last_frame.repeat(1, 1, missing, 1, 1))
                break
            if value.ndim != 5 or value.shape[:2] != (1, 3) or not value.shape[2]:
                raise ValueError("Input chunks must be nonempty [1,3,F,H,W] tensors")
            if self.last_frame is not None and value.shape[-2:] != self.last_frame.shape[-2:]:
                raise ValueError("Input chunk dimensions changed")
            first = self.source_frames == 0
            self.source_frames += value.shape[2]
            self.last_frame = value[:, :, -1:].clone()
            if first:
                expanded = torch.cat((value[:, :, :1], value[:, :, 1:].repeat_interleave(self.repeat_factor, dim=2)), dim=2)
            else:
                expanded = value.repeat_interleave(self.repeat_factor, dim=2)
            self._append(expanded)
        return self.expanded_frames >= stop

    def window(self, start: int, stop: int) -> torch.Tensor:
        self.ensure(stop)
        if start < self.offset or stop <= start:
            raise ValueError("Requested frames have already been released or window is empty")
        value = self.buffer[:, :, start - self.offset:stop - self.offset]
        if not value.shape[2]:
            value = self.last_frame[:, :, :0]
        missing = stop - start - value.shape[2]
        if missing:
            value = torch.cat((value, self.last_frame.repeat(1, 1, missing, 1, 1)), dim=2)
        return value.clone()

    def release_before(self, frame: int) -> None:
        frame = min(frame, self.expanded_frames)
        if frame > self.offset:
            self.buffer = self.buffer[:, :, frame - self.offset:].clone()
            self.offset = frame

    def projection_chunks(self, chunk_latents: int, model_compression: int):
        index = 0
        stride = chunk_latents * model_compression
        while self.ensure(1 + index * stride):
            start = 0 if index == 0 else 1 + index * stride
            stop = 1 + (index + 1) * stride
            value = self.window(start, stop)
            last = self.eof and self.expanded_frames < stop
            yield index, value, start, last
            index += 1

    def close(self) -> None:
        close = getattr(self.chunks, "close", None)
        if close is not None:
            close()
        self.buffer = None
        self.last_frame = None
