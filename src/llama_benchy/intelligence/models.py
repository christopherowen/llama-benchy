from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IntelligenceTaskResult(BaseModel):
    name: str = Field(..., description="Task name")
    metric: str = Field(..., description="Primary metric name")
    value: Optional[float] = Field(None, description="Primary metric value")
    duration_seconds: Optional[float] = Field(None, description="Task runtime in seconds")
    raw: Dict[str, Any] = Field(default_factory=dict, description="Raw framework output")


class IntelligencePluginResult(BaseModel):
    plugin: str = Field(..., description="Plugin name")
    success: bool = Field(..., description="Whether plugin finished successfully")
    summary_metric: Optional[float] = Field(None, description="Optional plugin-level summary metric")
    duration_seconds: Optional[float] = Field(None, description="Plugin runtime in seconds")
    artifacts: Dict[str, str] = Field(default_factory=dict, description="Plugin artifact paths")
    tasks: List[IntelligenceTaskResult] = Field(default_factory=list, description="Task-level results")
    error: Optional[str] = Field(None, description="Error text if plugin failed")


class IntelligenceReport(BaseModel):
    plugins: List[IntelligencePluginResult] = Field(default_factory=list, description="All intelligence plugin results")

