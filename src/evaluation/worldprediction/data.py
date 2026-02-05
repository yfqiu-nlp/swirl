"""Utilities for loading the WorldPrediction procedural planning metadata."""


import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class WorldPredictionActionStep:
    """Metadata describing a single action segment in the WorldPrediction dataset."""

    segment_uid: str
    video: str
    segment_start_time: float
    segment_end_time: float
    action_label: str
    action_uid: Optional[str] = None

    def representative_timestamp(self) -> float:
        """Return a timestamp inside the segment suitable for thumbnail extraction."""
        if not math.isfinite(self.segment_start_time):
            return 0.0
        if not math.isfinite(self.segment_end_time) or self.segment_end_time <= self.segment_start_time:
            return max(self.segment_start_time, 0.0)
        return max((self.segment_start_time + self.segment_end_time) * 0.5, 0.0)

    @classmethod
    def from_dict(cls, segment_uid: str, payload: Dict[str, object]) -> "WorldPredictionActionStep":
        try:
            video = str(payload.get("video", ""))
            start = float(payload.get("segment_start_time", 0.0))
            end = float(payload.get("segment_end_time", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed segment timings for {segment_uid}: {payload}") from exc

        action_label = str(payload.get("action_label", ""))
        action_uid = payload.get("action_uid")

        return cls(
            segment_uid=segment_uid,
            video=video,
            segment_start_time=start,
            segment_end_time=end,
            action_label=action_label,
            action_uid=str(action_uid) if action_uid is not None else None,
        )


@dataclass
class WorldPredictionSample:
    """Container for a single procedural planning instance."""

    dataset: str
    sample_uid: str
    ground_truth_index: int
    candidate_segments: List[List[str]]
    action_segments: Dict[str, WorldPredictionActionStep]
    original_segments: List[WorldPredictionActionStep]

    def __post_init__(self) -> None:
        if self.ground_truth_index < 0 or self.ground_truth_index >= len(self.candidate_segments):
            object.__setattr__(self, "ground_truth_index", max(0, min(self.ground_truth_index, len(self.candidate_segments) - 1)))

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_segments)

    @property
    def num_steps(self) -> int:
        if self.candidate_segments:
            return len(self.candidate_segments[0])
        return len(self.original_segments)

    @property
    def initial_segment_uid(self) -> Optional[str]:
        return self.original_segments[0].segment_uid if self.original_segments else None

    @property
    def target_segment_uid(self) -> Optional[str]:
        return self.original_segments[-1].segment_uid if self.original_segments else None

    def candidate_action_labels(self, candidate_index: int) -> List[str]:
        segment_ids = self.candidate_segments[candidate_index]
        labels: List[str] = []
        for segment_uid in segment_ids:
            action = self.action_segments.get(segment_uid)
            labels.append(action.action_label if action else "")
        return labels

    def candidate_action_steps(self, candidate_index: int) -> List[WorldPredictionActionStep]:
        segment_ids = self.candidate_segments[candidate_index]
        steps: List[WorldPredictionActionStep] = []
        for segment_uid in segment_ids:
            action = self.action_segments.get(segment_uid)
            if action is not None:
                steps.append(action)
        return steps


def _iter_samples_from_mapping(
    dataset_name: str,
    payload: Dict[str, object],
) -> Iterator[WorldPredictionSample]:
    for sample_uid, sample_payload in payload.items():
        if not isinstance(sample_payload, dict):
            continue

        states = sample_payload.get("states", {})
        sample_id = str(states.get("segment_uid", sample_uid))

        ground_truth_raw = sample_payload.get("ground_truth", 0)
        try:
            ground_truth_idx = int(ground_truth_raw)
        except (TypeError, ValueError):
            ground_truth_idx = 0

        if ground_truth_idx > 0:
            ground_truth_idx -= 1

        candidate_segments = [list(candidate) for candidate in sample_payload.get("candidates", []) if isinstance(candidate, Sequence)]

        action_segments_map: Dict[str, WorldPredictionActionStep] = {}
        action_segments_payload = sample_payload.get("action_segments", {})
        if isinstance(action_segments_payload, dict):
            for seg_uid, seg_payload in action_segments_payload.items():
                if not isinstance(seg_payload, dict):
                    continue
                try:
                    action_segments_map[seg_uid] = WorldPredictionActionStep.from_dict(seg_uid, seg_payload)
                except ValueError:
                    continue

        original_segments_payload = sample_payload.get("original_segments", [])
        original_steps: List[WorldPredictionActionStep] = []
        if isinstance(original_segments_payload, Sequence):
            for step_payload in original_segments_payload:
                if not isinstance(step_payload, dict):
                    continue
                seg_uid = str(step_payload.get("segment_uid", ""))
                if not seg_uid:
                    continue
                try:
                    original_steps.append(WorldPredictionActionStep.from_dict(seg_uid, step_payload))
                except ValueError:
                    continue

        yield WorldPredictionSample(
            dataset=dataset_name,
            sample_uid=sample_id,
            ground_truth_index=ground_truth_idx,
            candidate_segments=candidate_segments,
            action_segments=action_segments_map,
            original_segments=original_steps,
        )


def load_worldprediction_metadata(
    metadata_path: Path | str,
    dataset_filter: Optional[Sequence[str]] = None,
    sample_limit: Optional[int] = None,
) -> List[WorldPredictionSample]:
    """Load WorldPrediction procedural planning metadata.

    Args:
        metadata_path: Path to a JSON file with the structure described in the prompt.
        dataset_filter: Optional list of dataset/domain names to keep (e.g. ["COIN"]).
        sample_limit: Optional limit on number of samples to return (after filtering).

    Returns:
        A list of parsed :class:`WorldPredictionSample` instances.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"WorldPrediction metadata not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)

    datasets_to_keep: Optional[set[str]] = set(dataset_filter) if dataset_filter else None

    samples: List[WorldPredictionSample] = []

    def _accept(dataset_name: str) -> bool:
        return datasets_to_keep is None or dataset_name in datasets_to_keep

    if isinstance(raw_payload, dict):
        for key, value in raw_payload.items():
            if isinstance(value, dict) and any(isinstance(v, dict) and "states" in v for v in value.values()):
                dataset_name = str(key)
                if not _accept(dataset_name):
                    continue
                samples.extend(list(_iter_samples_from_mapping(dataset_name, value)))
            elif isinstance(value, dict) and "states" in value:
                dataset_name = str(value.get("dataset", key))
                if not _accept(dataset_name):
                    continue
                dummy_mapping = {key: value}
                samples.extend(list(_iter_samples_from_mapping(dataset_name, dummy_mapping)))
            else:
                dataset_name = str(key)
                if not _accept(dataset_name):
                    continue
                samples.extend(list(_iter_samples_from_mapping(dataset_name, {key: value})))
    elif isinstance(raw_payload, Sequence):
        dataset_name = "worldprediction"
        if _accept(dataset_name):
            for idx, sample_payload in enumerate(raw_payload):
                if not isinstance(sample_payload, dict):
                    continue
                pseudo_key = sample_payload.get("states", {}).get("segment_uid", f"sample_{idx:05d}")
                samples.extend(list(_iter_samples_from_mapping(dataset_name, {str(pseudo_key): sample_payload})))
    else:
        raise ValueError(f"Unsupported metadata format at {metadata_path}")

    if sample_limit is not None:
        samples = samples[:sample_limit]

    return samples
