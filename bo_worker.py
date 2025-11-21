from __future__ import annotations

import threading
import queue
from itertools import count
from typing import Dict, List, Optional

from skopt import gp_minimize
from skopt.space import Real

from common_types import StimJob, StimResult, StimParameters


MUSCLES = ("biceps_r", "triceps_r", "biceps_l", "triceps_l")
PARAMS_PER_MUSCLE = (
    ("onset_deg", 0.0, 360.0),
    ("offset_deg", 0.0, 360.0),
    ("pulse_intensity", 0.0, 10),
    ("pulse_width", 100.0, 500.0),
)


def build_search_space() -> List[Real]:
    """
    Create skopt search space: 4 parameters × 4 muscles = 16 dimensions.
    """
    space: List[Real] = []
    for muscle in MUSCLES:
        for param_name, low, high in PARAMS_PER_MUSCLE:
            dim_name = f"{param_name}_{muscle}"
            space.append(Real(low, high, name=dim_name))
    return space


class BayesianOptimizationWorker(threading.Thread):
    """
    Thread that runs Bayesian optimization and requests trials from
    the stimulation worker via a queue.

    Stimulation is continuous; each BO evaluation is:
      - send new StimParameters
      - wait for cost from stim thread
    """

    def __init__(
        self,
        job_queue: "queue.Queue[Optional[StimJob]]",
        stop_event: threading.Event,
        space: List[Real],
        name: str = "BayesianOptimizationWorker",
    ):
        super().__init__(name=name, daemon=True)
        self.job_queue = job_queue
        self.stop_event = stop_event
        self.space = space

        self._job_id_counter = count(start=1)
        self._results: Dict[int, StimResult] = {}
        self._result_lock = threading.Lock()
        self._result_available = threading.Condition(self._result_lock)

        self.best_result = None  # will hold gp_minimize's result

    # Called by StimulationWorker when a result is ready
    def handle_result(self, result: StimResult) -> None:
        with self._result_lock:
            self._results[result.job_id] = result
            self._result_available.notify_all()

    def _submit_job_and_wait_for_cost(self, params: StimParameters) -> float:
        job_id = next(self._job_id_counter)
        job = StimJob(job_id=job_id, params=params)

        self.job_queue.put(job)

        with self._result_lock:
            while job_id not in self._results:
                self._result_available.wait()
            result = self._results.pop(job_id)

        return float(result.cost)

    def _objective(self, x: List[float]) -> float:
        """
        Objective passed to gp_minimize.
        x is a flat vector of stimulation parameters.
        """
        if self.stop_event.is_set():
            print("[BO] Stop requested, returning large cost.")
            return 1e6

        params = StimParameters.from_flat_vector(x)
        cost = self._submit_job_and_wait_for_cost(params)
        print(f"[BO] Evaluated x={x} -> cost={cost}")
        return cost

    def run(self) -> None:
        """
        Main BO routine. Runs in a separate thread.
        """
        print("[BO] Starting Bayesian optimization with continuous stimulation...")

        self.best_result = gp_minimize(
            func=self._objective,
            dimensions=self.space,
            n_calls=6,
            n_initial_points=6,
            acq_func="EI",
            random_state=0,
        )

        print("[BO] Optimization finished.")
        print("[BO] Best parameters (flat vector):", self.best_result.x)
        print("[BO] Best cost:", self.best_result.fun)
