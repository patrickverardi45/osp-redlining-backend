from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


IssueType = Literal[
    "route_gap",
    "bore_offset",
    "deviation",
    "obstruction",
    "utility_conflict",
    "changed_bore_path",
    "new_conduit",
    "rod_and_rope",
    "strand_change",
    "handhole_change",
    "splice_change",
    "panel_change",
    "other",
]

GeometryType = Literal["point", "polyline", "polygon", "arrow", "cloud"]
SeverityType = Literal["info", "review", "alert"]
SourceType = Literal["foreman", "qa_engine", "bore_log", "kmz_compare"]


class GeometryPointModel(BaseModel):
    x: float
    y: float


class MarkupGeometryModel(BaseModel):
    geometry_type: GeometryType
    points: List[GeometryPointModel]
    color: str = "red"
    stroke_width: int = 3


class AssetChangeModel(BaseModel):
    asset_type: str
    asset_id: Optional[str] = None
    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)


class AsBuiltFieldsModel(BaseModel):
    owner: Optional[str] = None
    conduit_size: Optional[str] = None
    conduit_count: Optional[int] = None
    fiber_count: Optional[int] = None
    placement: Optional[str] = None
    sequential_in_ft: Optional[float] = None
    sequential_out_ft: Optional[float] = None
    panel_size: Optional[str] = None
    panel_assignment: Optional[str] = None
    pole_owner: Optional[str] = None
    note_text: Optional[str] = None


class RedlineRecordModel(BaseModel):
    record_id: str
    sheet_number: str
    station_from: Optional[str] = None
    station_to: Optional[str] = None
    issue_type: IssueType
    note_text: str
    geometry: MarkupGeometryModel
    affected_assets: List[AssetChangeModel] = Field(default_factory=list)
    as_built: AsBuiltFieldsModel = Field(default_factory=AsBuiltFieldsModel)
    source: SourceType = "qa_engine"
    severity: SeverityType = "review"
    metric_feet: Optional[float] = None
    verified_by_foreman: bool = False


class RedlineBatchModel(BaseModel):
    records: List[RedlineRecordModel]