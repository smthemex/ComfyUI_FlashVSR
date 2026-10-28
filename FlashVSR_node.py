 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import math
import weakref
import folder_paths
from comfy_api.latest import  io, ui, Types
from comfy_api.input_impl import VideoFromFile

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_FlashVSR_current_path = os.path.join(folder_paths.models_dir, "FlashVSR")
if not os.path.exists(weigths_FlashVSR_current_path):
    os.makedirs(weigths_FlashVSR_current_path)

folder_paths.add_model_folder_path("FlashVSR", weigths_FlashVSR_current_path) #  FlashVSR dir


# ---- 切分预览缓存：同一 trimmed 视频对象复用同一段临时预览文件，避免重复转码 ----
_trim_preview_results: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _save_trim_preview(video) -> ui.PreviewVideo:
    """把迭代切分出的视频段落写入临时目录并返回预览（参考官方 VideoTrim 的 save_video_preview）"""
    cached = _trim_preview_results.get(video)
    if cached is not None and os.path.isfile(cached[0]):
        return ui.PreviewVideo([cached[1]])

    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        "FlashVSR_temp_video", folder_paths.get_temp_directory(), 0, 0
    )
    file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(Types.VideoContainer.MP4)}"
    full_path = os.path.join(full_output_folder, file)
    video.save_to(
        full_path,
        format=Types.VideoContainer.MP4,
        codec="auto",
        preset="ultrafast",
    )
    result = ui.SavedResult(file, subfolder, io.FolderType.temp)
    _trim_preview_results[video] = (full_path, result)
    return ui.PreviewVideo([result])


class FlashVSR_SM_VideoTrimLoop(io.ComfyNode):
    """
    可视化视频循环切分节点（参考官方 LoadVideo + VideoTrim）。

    直接在节点上选择一个完整视频文件（无需预先切分），
    每次迭代按 split_length（秒）输出下一段视频，输出到最后一段后从头循环。
    """
    @classmethod
    def __init__(cls):
        cls.counters = {}

    @classmethod
    def define_schema(cls):
        # 与官方 LoadVideo 一致：列出 input 目录下可播放的视频文件，并支持上传按钮
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="FlashVSR_SM_VideoTrimLoop",
            display_name="FlashVSR_SM_VideoTrimLoop",
            category="FlashVSR",
            has_intermediate_output=True,
            inputs=[
                io.Combo.Input("video_file", options=sorted(files), upload=io.UploadType.video,
                               tooltip="选择完整视频文件，无需预先切分"),
                io.Float.Input("split_length", default=5.0, min=0.1, max=3600.0, step=0.1,
                               tooltip="每次迭代输出的切分时长（秒）"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("reset_bool", default=0, optional=True,
                             tooltip="输入值 >= 1 时，迭代计数器归零（从头开始）"),
            ],
            outputs=[
                io.Video.Output(),
                io.Custom("NUMBER").Output(display_name="number"),
                io.Int.Output(display_name="seed"),
                io.String.Output(display_name="filename"),
                io.Float.Output(display_name="start_time"),
            ],
        )

    @classmethod
    def execute(cls, video_file, split_length, seed, reset_bool=0) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(video_file)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Invalid video file: {video_file}")

        # ---- 迭代计数器：以文件绝对路径为 key ----
        UNIQUE_ID = os.path.normpath(video_path)
        start = 0
        step = 1
        counter = start
        if cls.counters.__contains__(UNIQUE_ID):
            counter = cls.counters[UNIQUE_ID]
        if round(reset_bool) >= 1:
            counter = start
        counter += step
        cls.counters[UNIQUE_ID] = counter
        result = int(counter)

        source = VideoFromFile(video_path)
        total_duration = float(source.get_duration())
        split_length = max(float(split_length), 0.1)
        num_segments = max(1, math.ceil(total_duration / split_length))

        # ---- 本次迭代对应的切分段：超出总段数后回到第 1 段（循环） ----
        segment_index = (result - 1) % num_segments
        start_time = segment_index * split_length
        seg_duration = min(split_length, max(total_duration - start_time, 0.1))

        trimmed = source.as_trimmed(start_time, seg_duration, strict_duration=False)
        if trimmed is None:
            raise ValueError(
                f"Failed to trim video:\nSource duration: {total_duration}\n"
                f"Start time: {start_time}\nTarget duration: {seg_duration}"
            )

        print(f"[FlashVSR_SM_VideoTrimLoop] segment {segment_index + 1}/{num_segments}: "
              f"start={start_time:.3f}s duration={seg_duration:.3f}s ({video_file})")

        filename = os.path.basename(video_path)
        return io.NodeOutput(
            trimmed, result, seed, filename, float(start_time),
            ui=_save_trim_preview(trimmed),
        )

    @classmethod
    def validate_inputs(cls, video_file, **kwargs):
        if not folder_paths.exists_annotated_filepath(video_file):
            return "Invalid video file: {}".format(video_file)
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # 每次入队都重新执行，保证迭代计数器推进
        return ""





