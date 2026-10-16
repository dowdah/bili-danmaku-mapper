"""CPU-only video sequence alignment and danmaku time mapping."""
from .config import AlignConfig
from .models import AlignmentResult, MappingResult


def align_videos(source, target, config=None, anchors=None):
    from .alignment import align_videos as run
    return run(source, target, config, anchors)


def map_danmaku(data, alignment, mode="strict"):
    from .mapping import map_danmaku as run
    return run(data, alignment, mode)


__all__ = ["AlignConfig", "AlignmentResult", "MappingResult", "align_videos", "map_danmaku"]
