from dataclasses import asdict, dataclass, field
import math


@dataclass
class Anchor:
    source_time: float
    target_time: float
    confidence: float = 1.0
    manual: bool = False
    evidence: str = "visual"


@dataclass
class Segment:
    source_start: float
    source_end: float
    target_start: float
    target_end: float
    slope: float
    offset: float
    confidence: float
    anchors: list[Anchor] = field(default_factory=list)


@dataclass
class AlignmentResult:
    source: str
    target: str
    source_duration: float
    target_duration: float
    segments: list[Segment]
    config: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Unsupported alignment schema_version")
        try:
            segments = [Segment(**{**s, "anchors": [Anchor(**a) for a in s["anchors"]]}) for s in data["segments"]]
            result = cls(**{**data, "segments": segments})
            result.validate()
            return result
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Malformed alignment document") from error

    def validate(self):
        if not all(math.isfinite(d) and d > 0 for d in (self.source_duration, self.target_duration)):
            raise ValueError("Invalid video duration")
        previous_s = previous_t = -1.0
        gap = self.config.get("strict_gap", 1.0)
        if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0:
            raise ValueError("Invalid strict interpolation gap")
        intervals = self.diagnostics.get("unverified_intervals", [])
        if not isinstance(intervals, list):
            raise ValueError("Invalid unverified interval list")
        for interval in intervals:
            if (not isinstance(interval, dict) or not isinstance(interval.get("reason"), str)
                    or not all(isinstance(interval.get(k), (float, int)) and math.isfinite(interval[k])
                               for k in ("source_start", "source_end"))
                    or not 0 <= interval["source_start"] <= interval["source_end"] <= self.source_duration):
                raise ValueError("Invalid unverified interval")
        for segment in self.segments:
            if not segment.anchors:
                raise ValueError("A segment requires anchors")
            for a in segment.anchors:
                if not all(math.isfinite(v) for v in (a.source_time, a.target_time, a.confidence)):
                    raise ValueError("Non-finite anchor")
                if not (0 <= a.source_time <= self.source_duration and 0 <= a.target_time <= self.target_duration):
                    raise ValueError("Anchor outside video duration")
                if a.source_time <= previous_s or a.target_time <= previous_t:
                    raise ValueError("Anchors must increase strictly in both timelines")
                if not 0 <= a.confidence <= 1:
                    raise ValueError("Invalid confidence")
                if not isinstance(a.manual, bool) or a.evidence not in ("visual", "audio", "manual"):
                    raise ValueError("Invalid anchor evidence")
                previous_s, previous_t = a.source_time, a.target_time
            if (segment.source_start != segment.anchors[0].source_time or
                    segment.source_end != segment.anchors[-1].source_time or
                    segment.target_start != segment.anchors[0].target_time or
                    segment.target_end != segment.anchors[-1].target_time):
                raise ValueError("Segment bounds must equal its anchor bounds")
            if (not all(math.isfinite(v) for v in (segment.slope, segment.offset, segment.confidence))
                    or segment.slope <= 0 or not 0 <= segment.confidence <= 1):
                raise ValueError("Invalid segment model")


@dataclass
class MappingResult:
    danmaku: dict
    review: dict
    report: dict
