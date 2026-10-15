from __future__ import annotations

import torch


def target_frame_count(frames: int, temporal_factor: int) -> int:
    if frames <= 0 or temporal_factor <= 0:
        raise ValueError("frames and temporal_factor must be positive")
    return ((frames - 1 + temporal_factor - 1) // temporal_factor) * temporal_factor + 1


def valid_frame_count(frames: int, temporal_factor: int) -> int:
    """Largest ``temporal_factor * n + 1`` frame count not exceeding input."""
    if frames <= 0 or temporal_factor <= 0:
        raise ValueError("frames and temporal_factor must be positive")
    return ((frames - 1) // temporal_factor) * temporal_factor + 1


def truncate_video_frames(
    video: torch.Tensor, temporal_factor: int
) -> tuple[torch.Tensor, int]:
    count = valid_frame_count(video.shape[2], temporal_factor)
    return video[:, :, :count], count


def repeat_last_frame(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    if target_frames < video.shape[2]:
        return video[:, :, :target_frames]
    missing = target_frames - video.shape[2]
    if missing == 0:
        return video
    return torch.cat(
        [video, video[:, :, -1:].repeat(1, 1, missing, 1, 1)], dim=2
    )


def pad_video_frames(video: torch.Tensor, temporal_factor: int, mode: str) -> tuple[torch.Tensor, int]:
    original = video.shape[2]
    target = target_frame_count(original, temporal_factor)
    missing = target - original
    if missing == 0:
        return video, original
    if mode == "repeat":
        padding = video[:, :, -1:].repeat(1, 1, missing, 1, 1)
    elif mode == "pingpong":
        if original == 1:
            padding = video.repeat(1, 1, missing, 1, 1)
        else:
            indices = list(range(original - 2, -1, -1))
            indices = (indices * ((missing + len(indices) - 1) // len(indices)))[:missing]
            padding = video[:, :, indices]
    else:
        raise ValueError(f"Unknown padding mode: {mode}")
    return torch.cat([video, padding], dim=2), original
