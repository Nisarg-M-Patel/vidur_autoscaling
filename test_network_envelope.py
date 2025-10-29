import math
import sys
import types
import unittest
from pathlib import Path
from typing import Iterable, List, Tuple

# Ensure the project root is on the path for direct test execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Stub the optional wandb dependency that pulls in heavy transitive imports when
# ``vidur.autoscaler`` is imported during unit tests.  The tests exercise the
# network envelope logic exclusively, so replacing the module with a lightweight
# stand-in keeps the import fast and avoids requiring wandb to be installed.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from vidur.autoscaler.inferline_autoscaler import NetworkEnvelope
from vidur.config.config import InferlineAutoscalerConfig
from vidur.entities.request import Request


def _make_request(arrival_time: float, tokens: int) -> Request:
    """Create a Request with a simple prefill/decode split for tests."""
    prefill = tokens // 2
    decode = tokens - prefill
    return Request(
        arrived_at=arrival_time,
        num_prefill_tokens=prefill,
        num_decode_tokens=decode,
    )


def _brute_force_max_rate(
    arrivals: Iterable[Tuple[float, int]],
    time: float,
    window_size: float,
    look_back_time: float,
) -> float:
    """Compute the theoretical max token rate for comparison."""
    if window_size <= 0:
        return 0.0

    cutoff = time - look_back_time
    relevant: List[Tuple[float, int]] = [
        (arrival_time, tokens)
        for arrival_time, tokens in arrivals
        if cutoff <= arrival_time <= time
    ]
    if not relevant:
        return 0.0

    candidate_starts = {cutoff}
    for arrival_time, _ in relevant:
        candidate_starts.add(arrival_time)
        candidate_starts.add(max(cutoff, arrival_time - window_size))

    best_rate = 0.0
    for window_start in sorted(candidate_starts):
        window_end = min(window_start + window_size, time)
        duration = window_end - window_start
        if duration <= 0:
            continue

        tokens_in_window = sum(
            tokens
            for arrival_time, tokens in relevant
            if window_start <= arrival_time < window_end
        )
        rate = tokens_in_window / duration
        best_rate = max(best_rate, rate)

    return best_rate


class TestNetworkEnvelope(unittest.TestCase):
    """Unit tests for the InferLine network envelope implementation."""

    def setUp(self) -> None:
        self.autoscaler_config = InferlineAutoscalerConfig(
            look_back_time_scale_up=12,
            look_back_time_scale_down=12,
            min_window_size_scale_up=4,
            min_window_size_scale_down=4,
            stabilization_delay=5,
            initial_replica_token_throughput=1.0,
        )
        self.envelope = NetworkEnvelope(self.autoscaler_config)

    def test_arrivals_outside_lookback_are_dropped(self):
        """Old arrivals should be purged before computing the max rate."""
        arrivals = [
            (4.0, 60),
            (7.2, 45),
            (8.1, 50),
            (9.7, 80),
        ]

        for arrival_time, tokens in arrivals:
            self.envelope.on_request_arrival(_make_request(arrival_time, tokens))

        time = 12.0
        window_size = 4.0
        look_back_time = 6.0

        computed_rate = self.envelope.get_max_request_rate(
            time=time,
            window_size=window_size,
            look_back_time=look_back_time,
        )
        expected_rate = _brute_force_max_rate(
            arrivals,
            time=time,
            window_size=window_size,
            look_back_time=look_back_time,
        )

        retained_arrivals = list(self.envelope._arrivals)
        print("\nRetained arrivals after look-back trim:", retained_arrivals)
        print("Computed max rate:", computed_rate)
        print("Expected max rate:", expected_rate)

        cutoff = time - look_back_time
        self.assertTrue(all(arrival_time >= cutoff for arrival_time, _ in retained_arrivals))
        self.assertTrue(math.isclose(computed_rate, expected_rate, rel_tol=1e-9))

    def test_repeated_queries_trim_processed_arrivals(self):
        """Multiple invocations should continue trimming old arrivals."""
        arrivals = [
            (2.0, 40),
            (7.5, 80),
            (9.0, 60),
        ]

        for arrival_time, tokens in arrivals:
            self.envelope.on_request_arrival(_make_request(arrival_time, tokens))

        first_time = 12.0
        second_time = 30.0
        window_size = 4.0
        look_back_time = 6.0

        first_rate = self.envelope.get_max_request_rate(
            time=first_time,
            window_size=window_size,
            look_back_time=look_back_time,
        )
        print("\nFirst rate:", first_rate)
        print("Arrivals after first query:", list(self.envelope._arrivals))

        second_rate = self.envelope.get_max_request_rate(
            time=second_time,
            window_size=window_size,
            look_back_time=look_back_time,
        )
        print("Second rate:", second_rate)
        print("Arrivals after second query:", list(self.envelope._arrivals))

        self.assertGreater(first_rate, 0.0)
        self.assertEqual(second_rate, 0.0)
        self.assertEqual(len(self.envelope._arrivals), 0)

    def test_dense_arrival_burst_matches_bruteforce(self):
        """Dense arrivals should match a brute-force sliding window calculation."""
        arrivals = [
            (10.0, 60),
            (10.5, 30),
            (11.0, 45),
            (11.4, 30),
        ]

        for arrival_time, tokens in arrivals:
            self.envelope.on_request_arrival(_make_request(arrival_time, tokens))

        time = 12.0
        window_size = 1.0
        look_back_time = 4.0

        computed_rate = self.envelope.get_max_request_rate(
            time=time,
            window_size=window_size,
            look_back_time=look_back_time,
        )
        expected_rate = _brute_force_max_rate(
            arrivals,
            time=time,
            window_size=window_size,
            look_back_time=look_back_time,
        )

        print("\nDense arrival series:", list(self.envelope._arrivals))
        print("Computed dense burst rate:", computed_rate)
        print("Expected dense burst rate:", expected_rate)

        self.assertTrue(math.isclose(computed_rate, expected_rate, rel_tol=1e-9))

    def test_zero_window_returns_zero_rate(self):
        """Zero window size should return zero without raising errors."""
        rate = self.envelope.get_max_request_rate(
            time=5.0,
            window_size=0.0,
            look_back_time=5.0,
        )
        print("\nZero window rate:", rate)
        self.assertEqual(rate, 0.0)


if __name__ == "__main__":
    unittest.main()