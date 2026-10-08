from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class HourlyTelemetryRecord(BaseModel):
    timestamp: datetime
    prb_util_mean: float
    traffic_mbps_mean: float
    sinr_mean: float
    bler_mean: float
    drop_rate_mean: float
    availability_mean: float
    handover_success_mean: float
    alarm_count: int = 0


class StaticAttributes(BaseModel):
    technology: str
    band: str
    site_type: str
    region_name: str
    cell_capacity_mbps: float


class FaultDetectionRequest(BaseModel):
    cell_id: str
    static: StaticAttributes
    telemetry_window: List[HourlyTelemetryRecord] = Field(
        ..., min_length=2, description="Hourly records, oldest first, at least 5 recommended for full feature coverage")


class FaultDetectionResponse(BaseModel):
    cell_id: str
    timestamp: datetime
    fault_probability: float
    is_fault_flagged: bool
    threshold_used: float


class FeatureContribution(BaseModel):
    feature: str
    value: Optional[float]
    shap_value: float


class FaultExplanationResponse(FaultDetectionResponse):
    top_contributions: List[FeatureContribution]


class CongestionRequest(BaseModel):
    cell_id: str
    static: StaticAttributes
    telemetry_window: List[HourlyTelemetryRecord] = Field(
        ..., min_length=2,
        description="Longer history improves accuracy — up to 336 hourly rows (14 days) used if provided")


class CongestionResponse(BaseModel):
    cell_id: str
    timestamp: datetime
    horizon: str
    forecast_prb_utilization_pct: float
    history_hours_used: int


class AnomalyValuePoint(BaseModel):
    timestamp: datetime
    value: float


class AnomalyRequest(BaseModel):
    cell_id: str
    metric_name: str
    recent_values: List[AnomalyValuePoint] = Field(
        ..., min_length=1, description="Last 1-3 hourly values for this metric, most recent last")


class AnomalyResponse(BaseModel):
    cell_id: str
    metric_name: str
    timestamp: datetime
    actual_value: float
    expected_value: Optional[float]
    z_score: Optional[float]
    is_anomaly_flagged: bool
