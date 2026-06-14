"""Request/response contracts for the serving API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BiomarkerInput(BaseModel):
    """A biomarker panel. Defaults are the optimal midpoints, so partial
    payloads still produce a sensible prediction."""
    vitamin_d: float = Field(65.0, ge=0)
    ferritin: float = Field(110.0, ge=0)
    fasting_glucose: float = Field(82.0, ge=0)
    hba1c: float = Field(5.1, ge=0)
    tsh: float = Field(1.8, ge=0)
    hscrp: float = Field(0.6, ge=0)
    hdl: float = Field(70.0, ge=0)
    ldl: float = Field(85.0, ge=0)
    triglycerides: float = Field(75.0, ge=0)
    vitamin_b12: float = Field(650.0, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "vitamin_d": 32, "ferritin": 45, "fasting_glucose": 104,
                "hba1c": 5.9, "tsh": 2.1, "hscrp": 3.2, "hdl": 48,
                "ldl": 130, "triglycerides": 160, "vitamin_b12": 380,
            }
        }
    }


class PredictionResponse(BaseModel):
    biological_age: float
    model_type: str
    model_mae: float | None = None
