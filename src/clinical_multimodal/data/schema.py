from __future__ import annotations

from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Contract:
    vital_names: list[str]
    lab_names: list[str]
    static_names: list[str]
    lab_channels: list[str]
    normalization_hash: str
    schema_version: str = "tensor-v1"
    vital_age_unit: str = "minutes"
    lab_age_unit: str = "hours"
    note_age_unit: str = "hours"

    def to_dict(self):
        return asdict(self)

    def assert_matches(self, other: "Contract") -> None:
        if self != other:
            differences = [k for k, v in self.to_dict().items() if v != other.to_dict().get(k)]
            raise ValueError(f"Input contract mismatch (no silent reordering): {differences}")
