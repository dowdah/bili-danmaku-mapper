"""Configuration is resolved at call time, not import time."""
from dataclasses import asdict, dataclass
from pathlib import Path
import os

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ALGORITHM_VERSION = "sequence-5"
BASE_ALGORITHM_VERSION = "sequence-4"
FEATURE_VERSION = "sequence-2"


def boolean(value):
    value=str(value).lower()
    if value in ('1','true','yes','on'):return True
    if value in ('0','false','no','off'):return False
    raise ValueError('Boolean configuration must be true or false')


def settings(env_file=None):
    values = {k: v for k, v in dotenv_values(env_file or ROOT / ".env").items() if v is not None}
    values.update(os.environ)
    return values


@dataclass(frozen=True)
class AlignConfig:
    sample_rate: float = 2.0
    crop_ratio: float = 0.8
    top_k: int = 5
    context_seconds: float = 3.0
    max_hash_distance: int = 10
    strict_gap: float = 1.0
    max_segment_gap: float = 10.0
    refine: bool = True
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    hwaccel: str | None = None
    source_hwaccel: str | None = None
    target_hwaccel: str | None = None
    cache_dir: str | None = None
    algorithm: str = "sequence"
    evidence_mode: str = "hybrid"
    hybrid_strategy: str = "visual_first"
    audio_speed_search: bool = True
    alignment_cache: bool = True
    coverage_recovery: bool = False
    recovery_stage: str = "audio"
    dense_cache: bool = True
    merge_refinement: bool = True
    source_audio_stream: int | None = None
    target_audio_stream: int | None = None

    def __post_init__(self):
        import math
        for name in ("sample_rate", "crop_ratio", "context_seconds", "strict_gap", "max_segment_gap"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.sample_rate > 60 or self.crop_ratio > 1 or not 1 <= self.top_k <= 20:
            raise ValueError("sample_rate <= 60, crop_ratio <= 1 and 1 <= top_k <= 20 required")
        if not 0 <= self.max_hash_distance <= 64:
            raise ValueError("max_hash_distance must be between 0 and 64")
        if self.algorithm not in ("sequence", "legacy"):
            raise ValueError("algorithm must be sequence or legacy")
        if self.evidence_mode not in ("visual", "hybrid", "audio"):
            raise ValueError("evidence_mode must be visual, hybrid or audio")
        if self.recovery_stage not in ("fusion", "visual", "audio"):
            raise ValueError("recovery_stage must be fusion, visual or audio")
        if not isinstance(self.coverage_recovery, bool):
            raise ValueError("coverage_recovery must be boolean")
        if self.coverage_recovery and self.algorithm != "sequence":
            raise ValueError("Coverage recovery requires sequence alignment")
        if not isinstance(self.audio_speed_search,bool):
            raise ValueError("audio_speed_search must be boolean")
        if self.hybrid_strategy not in ("audio_first", "visual_first"):
            raise ValueError("hybrid_strategy must be audio_first or visual_first")
        for backend in (self.source_hwaccel, self.target_hwaccel):
            if backend not in (None, "software", "videotoolbox"):
                raise ValueError("Per-video backend must be software or videotoolbox")
        for stream in (self.source_audio_stream, self.target_audio_stream):
            if stream is not None and (not isinstance(stream, int) or isinstance(stream, bool) or stream < 0):
                raise ValueError("Audio stream selectors must be nonnegative audio-stream ordinals")

    @classmethod
    def resolve(cls, env_file=None, **overrides):
        env = settings(env_file)
        mapping = {
            "sample_rate": ("SAMPLE_RATE", float), "crop_ratio": ("CROP_RATIO", float),
            "top_k": ("TOP_K", int), "context_seconds": ("CONTEXT_SECONDS", float),
            "max_hash_distance": ("MAX_HAMMING_DISTANCE", int),
            "strict_gap": ("STRICT_GAP", float), "max_segment_gap": ("MAX_SEGMENT_GAP", float),
            "ffmpeg": ("FFMPEG_PATH", str), "ffprobe": ("FFPROBE_PATH", str),
            "hwaccel": ("HWACCEL", str), "algorithm": ("ALGORITHM", str),
            "source_hwaccel": ("SOURCE_HWACCEL", str), "target_hwaccel": ("TARGET_HWACCEL", str),
            "evidence_mode": ("EVIDENCE_MODE", str),
            "hybrid_strategy": ("HYBRID_STRATEGY", str),
            "audio_speed_search": ("AUDIO_SPEED_SEARCH", boolean),
            "coverage_recovery": ("COVERAGE_RECOVERY", boolean),
            "recovery_stage": ("RECOVERY_STAGE", str),
            "source_audio_stream": ("SOURCE_AUDIO_STREAM", int),
            "target_audio_stream": ("TARGET_AUDIO_STREAM", int),
        }
        values = {name: cast(env[key]) for name, (key, cast) in mapping.items() if env.get(key)}
        values["cache_dir"] = str(Path(env.get("BASE_DIR", ROOT / "data")) / "features")
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def feature_signature(self, sample_rate=None):
        return {"version": FEATURE_VERSION, "rate": self.sample_rate if sample_rate is None else sample_rate,
                "crop_ratio": self.crop_ratio, "size": [256, 144]}

    def for_video(self, role):
        from dataclasses import replace
        if role not in ("source", "target"):
            raise ValueError("Video role must be source or target")
        backend = getattr(self, role + "_hwaccel")
        return self if backend is None else replace(self, hwaccel=None if backend == "software" else backend)

    def to_dict(self):
        return asdict(self)
