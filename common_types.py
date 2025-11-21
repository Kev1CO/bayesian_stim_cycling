from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class StimParameters:
    """
    Single BO sample of stimulation parameters, matching your BayesianOptim:

      onset_deg / offset_deg / pulse_intensity / pulse_width
      for each muscle: biceps_r, triceps_r, biceps_l, triceps_l.
    """

    # Right biceps
    onset_deg_biceps_r: float
    offset_deg_biceps_r: float
    pulse_intensity_biceps_r: float
    pulse_width_biceps_r: float

    # Right triceps
    onset_deg_triceps_r: float
    offset_deg_triceps_r: float
    pulse_intensity_triceps_r: float
    pulse_width_triceps_r: float

    # Left biceps
    onset_deg_biceps_l: float
    offset_deg_biceps_l: float
    pulse_intensity_biceps_l: float
    pulse_width_biceps_l: float

    # Left triceps
    onset_deg_triceps_l: float
    offset_deg_triceps_l: float
    pulse_intensity_triceps_l: float
    pulse_width_triceps_l: float

    @classmethod
    def from_flat_vector(cls, x: List[float]) -> "StimParameters":
        """
        Convert 16D BO vector to StimParameters instance.
        Order must match the search space in bo_worker.py.
        """
        return cls(*x)

    def to_flat_vector(self) -> List[float]:
        """
        Convert back to a flat list if needed.
        """
        return [
            self.onset_deg_biceps_r,
            self.offset_deg_biceps_r,
            self.pulse_intensity_biceps_r,
            self.pulse_width_biceps_r,
            self.onset_deg_triceps_r,
            self.offset_deg_triceps_r,
            self.pulse_intensity_triceps_r,
            self.pulse_width_triceps_r,
            self.onset_deg_biceps_l,
            self.offset_deg_biceps_l,
            self.pulse_intensity_biceps_l,
            self.pulse_width_biceps_l,
            self.onset_deg_triceps_l,
            self.offset_deg_triceps_l,
            self.pulse_intensity_triceps_l,
            self.pulse_width_triceps_l,
        ]


@dataclass
class StimJob:
    """
    A job requested by the optimizer:
      - job_id: unique ID so we can match result to request
      - params: stimulation parameters to test on hardware
    """
    job_id: int
    params: StimParameters


@dataclass
class StimResult:
    """
    Result returned by the stimulation thread:
      - job_id: must match the StimJob
      - cost: scalar value to give back to the optimizer
      - extra_data: optional (e.g., raw sensor data, logs)
    """
    job_id: int
    cost: float
    extra_data: Optional[Dict] = None

