from __future__ import annotations

import threading
import queue
import time
from typing import Optional, Dict, Callable, List

import numpy as np
import nidaqmx

from pysciencemode import Rehastim2 as St
from pysciencemode import Channel as Ch
from pysciencemode import Device, Modes

from common_types import StimJob, StimResult, StimParameters


class HandCycling2:
    """
    Hardware controller for Rehastim2 and encoder.

    Stimulation is started once and kept running.
    Angle is read from a single NI-DAQ channel (e.g. Dev1/ai14).
    BO updates only the stimulation parameters.
    """

    MUSCLE_KEYS = ["biceps_r", "triceps_r", "biceps_l", "triceps_l"]

    def __init__(self):
        # ----------------- Stimulator setup ----------------- #
        channel_muscle_name = [
            "biceps_r",
            "triceps_r",
            "biceps_l",
            "triceps_l",
        ]
        self.list_channels = [
            Ch(
                mode=Modes.SINGLE,
                no_channel=i + 1,
                amplitude=0,
                pulse_width=350,
                name=channel_muscle_name[i],
                device_type=Device.Rehastim2,
            )
            for i in range(len(channel_muscle_name))
        ]

        # Default intensity for each muscle (will be overridden by BO)
        self.intensity = {
            "biceps_r": 10,
            "triceps_r": 10,
            "biceps_l": 10,
            "triceps_l": 10,
        }

        # Default pulse width for each muscle (will be overridden by BO)
        self.pulse_width = {
            "biceps_r": 100,
            "triceps_r": 100,
            "biceps_l": 100,
            "triceps_l": 100,
        }

        self.channel_number = {
            "biceps_r": 1,
            "triceps_r": 2,
            "biceps_l": 3,
            "triceps_l": 4,
        }

        self.stimulation_state = {
            "biceps_r": False,
            "triceps_r": False,
            "biceps_l": False,
            "triceps_l": False,
        }

        # Create stimulator
        self.stimulator = St(port="COM6", show_log=False)
        self.stimulator.init_channel(
            stimulation_interval=30,
            list_channels=self.list_channels,
        )

        # Default stimulation ranges in degrees (will be overridden by BO)
        self.stimulation_range = {
            "biceps_r": [220.0, 10.0],
            "triceps_r": [20.0, 180.0],
            "biceps_l": [40.0, 190.0],
            "triceps_l": [200.0, 360.0],
        }

        # Condition flags for wrap-around
        self.stim_condition: Dict[str, int] = {}
        self._update_stim_condition()

        # ----------------- Encoder setup ----------------- #
        local_system = nidaqmx.system.System.local()
        driver_version = local_system.driver_version
        print(
            "DAQmx {0}.{1}.{2}".format(
                driver_version.major_version,
                driver_version.minor_version,
                driver_version.update_version,
            )
        )
        for device in local_system.devices:
            print(
                "Device Name: {0}, Product Category: {1}, Product Type: {2}".format(
                    device.name, device.product_category, device.product_type
                )
            )
            device_mane = device.name
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(device_mane + "/ai14")
            self.task.start()

            self.min_voltage = 1.33
            max_voltage = 5
            self.origin = self.task.read() - self.min_voltage
            self.angle_coeff = 360 / (max_voltage - self.min_voltage)
            self.actual_voltage = None

        self.angle = 0.0

        # ----------------- Start stimulation once ----------------- #
        self.stimulator.start_stimulation(upd_list_channels=self.list_channels)


    # ------------ Apply parameters from BO (called only when BO updates) ------------ #
    def _update_stim_condition(self):
        for key in self.stimulation_range.keys():
            self.stim_condition[key] = (
                1 if self.stimulation_range[key][0] < self.stimulation_range[key][1] else 0
            )

    def apply_parameters(self, params: StimParameters) -> None:
        """
        Apply BO parameters to:
          - stimulation_range (onset/offset)
          - intensity per muscle
          - pulse_width per muscle
        """
        for muscle in self.MUSCLE_KEYS:
            onset = int(getattr(params, f"onset_deg_{muscle}"))
            offset = int(getattr(params, f"offset_deg_{muscle}"))
            intensity = int(getattr(params, f"pulse_intensity_{muscle}"))
            pulse_width = int(getattr(params, f"pulse_width_{muscle}"))

            # Update angle range [onset, offset]
            self.stimulation_range[muscle] = [onset, offset]

            # Update intensity & pulse width
            self.intensity[muscle] = intensity

            # Update pulse width
            self.pulse_width[muscle] = pulse_width

        # Recompute condition flags
        self._update_stim_condition()

    # ----------------- Angle reading ----------------- #
    def get_angle(self) -> float:
        """
        Real angle from encoder.
        """
        voltage = self.task.read() - self.min_voltage
        self.actual_voltage = voltage - self.origin
        self.angle = (
            360 - (self.actual_voltage * self.angle_coeff)
            if 0 < self.actual_voltage <= 5 - self.origin
            else abs(self.actual_voltage) * self.angle_coeff
        )
        return self.angle

    def fake_angle(self) -> float:
        """
        For testing without encoder: increase angle by 0.5° per loop and wrap at 360.
        """
        temp = self.angle + 0.5
        self.angle = temp % 360.0
        return self.angle

    # ----------------- Stimulation update ----------------- #
    def update_stimulation_for_current_angle(self) -> bool:
        """
        One pass of your original while-loop logic.

        Returns:
          True if amplitudes were updated and we should call start_stimulation().
        """
        stim_to_update = False
        for key in self.stimulation_range.keys():
            onset, offset = self.stimulation_range[key]
            cond = self.stim_condition[key]
            state = self.stimulation_state[key]
            ch_idx = self.channel_number[key] - 1
            channel = self.list_channels[ch_idx]

            if cond == 0:
                # Range wraps around 360 (e.g., [220, 10])
                if (
                    (onset <= self.angle <= 360.0)
                    or (0.0 <= self.angle <= offset)
                ) and not state:
                    self.stimulation_state[key] = True
                    channel.set_amplitude(self.intensity[key])
                    channel.set_pulse_width(self.pulse_width[key])
                    stim_to_update = True
                elif (offset < self.angle < onset) and state:
                    self.stimulation_state[key] = False
                    channel.set_amplitude(0.0)
                    stim_to_update = True
            else:
                # Simple range [onset, offset]
                if onset <= self.angle <= offset and not state:
                    self.stimulation_state[key] = True
                    channel.set_amplitude(self.intensity[key])
                    channel.set_pulse_width(self.pulse_width[key])
                    stim_to_update = True
                elif (self.angle < onset or self.angle > offset) and state:
                    self.stimulation_state[key] = False
                    channel.set_amplitude(0.0)
                    stim_to_update = True

        return stim_to_update


class StimulationWorker(threading.Thread):
    """
    Thread that:
      - keeps stimulation running continuously using a while-loop structure
      - whenever a new StimJob arrives, it:
          * applies the new parameters ONCE
          * collects data for an evaluation window (eval_duration_s)
          * computes a cost
          * sends StimResult back via callback

    Stimulation NEVER stops; only parameters and evaluation windows change.
    """

    def __init__(
        self,
        job_queue: "queue.Queue[Optional[StimJob]]",
        stop_event: threading.Event,
        result_callback: Callable[[StimResult], None],
        eval_duration_s: float = 2.0,
        name: str = "StimulationWorker",
    ):
        super().__init__(name=name, daemon=True)
        self.job_queue = job_queue
        self.stop_event = stop_event
        self.result_callback = result_callback
        self.eval_duration_s = eval_duration_s

        # Single hardware controller that runs continuously
        self.controller = HandCycling2()

        # State for current BO evaluation
        self.current_job: Optional[StimJob] = None
        self.eval_start_time: Optional[float] = None
        self.eval_buffer: List[Dict] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                while True:
                    job = self.job_queue.get_nowait()
                    if job is None:
                        self.job_queue.task_done()
                        return

                    # New BO evaluation request
                    self._start_new_evaluation(job)
                    self.job_queue.task_done()
            except queue.Empty:
                pass

            # Get the pedal angle
            # angle = self.controller.fake_angle()
            angle = self.controller.get_angle()
            # print(angle)

            if self.controller.update_stimulation_for_current_angle():
                # Only call when intensity or pulse width changed
                self.controller.stimulator.start_stimulation(
                    upd_list_channels=self.controller.list_channels
                )

            # Build a measurement dict
            measurement = {
                "time": time.time(),
                "angle": angle,
                # "torque": ...,
                # "force": ...,
            }

            if self.current_job is not None:
                self.eval_buffer.append(measurement)

                elapsed = time.time() - self.eval_start_time
                if elapsed >= self.eval_duration_s:
                    self._finish_current_evaluation()

            time.sleep(0.001)

    def _start_new_evaluation(self, job: StimJob) -> None:
        """
        Called when BO sends a new parameter set.
        """
        print(f"[StimulationWorker] Starting evaluation of job {job.job_id}")
        self.current_job = job
        self.eval_start_time = time.time()
        self.eval_buffer = []

        # Apply parameters immediately and stimulation continues with new params
        self.controller.apply_parameters(job.params)

    def _finish_current_evaluation(self) -> None:
        """
        Compute cost from buffered data and report back to BO.
        """
        assert self.current_job is not None
        job = self.current_job

        cost = self._compute_cost_from_buffer(self.eval_buffer)
        extra_data = {
            "num_samples": len(self.eval_buffer),
        }

        print(f"[StimulationWorker] Finished evaluation of job {job.job_id}")
        result = StimResult(job_id=job.job_id, cost=cost, extra_data=extra_data)
        self.result_callback(result)

        # Keep stimulation running with last parameters, just end evaluation
        self.current_job = None
        self.eval_start_time = None
        self.eval_buffer = []

    # TODO: Replace this with a real cost function.
    def _compute_cost_from_buffer(self, buffer: List[Dict]) -> float:
        """
        Compute a scalar cost from the collected data during eval_duration_s.
        For now, it uses a dummy example based on angle variance.
        """
        if not buffer:
            return 1.0  # something non-catastrophic

        angles = np.array([m["angle"] for m in buffer], dtype=float)

        # Dummy example cost:
        #  - smaller is better
        #  - here: we just take 1 / (1 + std(angle)) as a placeholder
        std_angle = float(np.std(angles))
        cost = 1.0 / (1.0 + std_angle)

        # Replace with something meaningful, e.g.:
        #   cost = tracking_error(force_signal, desired_force_profile)
        #   cost = -mean_power
        #   etc.
        return float(cost)
