"""Conservative disk admission and a fixed free-space floor for playback files."""
from pathlib import Path
import shutil

GIB = 1024**3
MIN_FREE = 2 * GIB


def check_space(directory, source_bytes=0, *, admission=True):
    # Work backwards to an existing parent before a task directory is created.
    destination = Path(directory)
    while not destination.exists():
        destination = destination.parent
    free = shutil.disk_usage(destination).free
    # CRF cannot guarantee a file size. Source size is an admission estimate,
    # while the runtime floor also protects against outputs exceeding it.
    reserve = max(MIN_FREE, int(source_bytes*.05)) if admission else MIN_FREE
    estimate = source_bytes if admission else 0
    if free < estimate + reserve:
        raise ValueError(f'播放准备空间不足：当前可用 {free/GIB:.1f} GiB，'
                         f'本阶段保守预留需 {(estimate+reserve)/GIB:.1f} GiB。'
                         '已停止生成播放文件；请先用原片对齐，或使用空间充足的磁盘。')
    return {'free_bytes':free,'estimated_output_bytes':estimate,'reserve_bytes':reserve}
