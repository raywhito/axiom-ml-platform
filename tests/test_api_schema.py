"""Tests for the API request contract."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import BiomarkerInput


def test_defaults_are_applied():
    payload = BiomarkerInput()
    assert payload.vitamin_d == 65.0
    assert payload.hba1c == 5.1


def test_partial_payload_ok():
    payload = BiomarkerInput(fasting_glucose=104, hba1c=5.9)
    assert payload.fasting_glucose == 104
    assert payload.ldl == 85.0  # default kept


def test_negative_value_rejected():
    with pytest.raises(ValidationError):
        BiomarkerInput(vitamin_d=-5)
