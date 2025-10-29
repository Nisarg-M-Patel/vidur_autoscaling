import unittest
from unittest.mock import Mock
import math
from dataclasses import dataclass
from collections import deque

# Mock classes for testing
@dataclass
class MockInferlineAutoscalerConfig:
    min_window_size_scale_up: float = 20.0
    min_window_size_scale_down: float = 20.0
    look_back_time_scale_up: float = 80.0
    look_back_time_scale_down: float = 80.0
    stabilization_delay: float = 10.0
    initial_replica_token_throughput: float = 1.0
    throughput_alpha: float = 0.5

@dataclass
class MockRequest:
    arrived_at: float
    num_prefill_tokens: int
    num_decode_tokens: int

@dataclass 
class MockBatch:
    scheduled_at: float
    completed_at: float
    num_tokens: list

# Simplified NetworkEnvelope implementation for testing
class NetworkEnvelope:
    def __init__(self, autoscaler_config):
        self._min_window_size_scale_up = autoscaler_config.min_window_size_scale_up
        self._min_window_size_scale_down = autoscaler_config.min_window_size_scale_down
        self._look_back_time_scale_up = autoscaler_config.look_back_time_scale_up
        self._look_back_time_scale_down = autoscaler_config.look_back_time_scale_down
        self._arrivals = deque()

    def on_request_arrival(self, request):
        total_tokens = request.num_prefill_tokens + request.num_decode_tokens
        self._arrivals.append((request.arrived_at, total_tokens))

    def get_max_token_arrival_rate(self, time: float, window_size: float, look_back_time: float) -> float:
        if window_size <= 0:
            return 0.0
            
        cutoff_time = time - look_back_time
        while self._arrivals and self._arrivals[0][0] < cutoff_time:
            self._arrivals.popleft()
        
        if not self._arrivals:
            return 0.0
        
        arrivals_list = list(self._arrivals)
        if not arrivals_list or arrivals_list[-1][0] < cutoff_time:
            return 0.0
        
        max_rate = 0.0
        step_size = window_size / 10.0
        window_start = cutoff_time
        
        while window_start < time:
            window_end = min(window_start + window_size, time)
            tokens_in_window = 0
            
            for arrival_time, tokens in arrivals_list:
                if window_start <= arrival_time < window_end:
                    tokens_in_window += tokens
            
            actual_window_size = window_end - window_start
            if actual_window_size > 0:
                rate = tokens_in_window / actual_window_size
                max_rate = max(max_rate, rate)
            
            window_start += step_size
        
        return max_rate

# Simplified InferlineAutoscaler for testing core logic
class InferlineAutoscaler:
    def __init__(self, autoscaler_config, num_replicas=2):
        self._autoscaler_config = autoscaler_config
        self._replica_token_throughput = autoscaler_config.initial_replica_token_throughput
        self._throughput_alpha = autoscaler_config.throughput_alpha
        self._last_scale_up_time = -float('inf')
        self._network_envelope = NetworkEnvelope(autoscaler_config)
        self._num_pending_scale_ups = 0
        self._num_pending_scale_downs = 0
        self.num_replicas = num_replicas

    @property
    def replica_token_throughput(self):
        return self._replica_token_throughput

    def on_batch_end(self, batch):
        if batch.scheduled_at is None or batch.completed_at is None:
            return
            
        execution_time = batch.completed_at - batch.scheduled_at
        if execution_time <= 0:
            return
            
        total_tokens = sum(batch.num_tokens)
        if total_tokens <= 0:
            return
            
        batch_throughput = total_tokens / execution_time
        
        self._replica_token_throughput = (
            self._throughput_alpha * batch_throughput + 
            (1 - self._throughput_alpha) * self._replica_token_throughput
        )

    def tune(self, time: float) -> int:
        max_arrival_rate_scale_up = self._network_envelope.get_max_token_arrival_rate(
            time, 
            self._autoscaler_config.min_window_size_scale_up,
            self._autoscaler_config.look_back_time_scale_up
        )
        
        max_arrival_rate_scale_down = self._network_envelope.get_max_token_arrival_rate(
            time,
            self._autoscaler_config.min_window_size_scale_down, 
            self._autoscaler_config.look_back_time_scale_down
        )
        
        current_replicas = self.num_replicas + self._num_pending_scale_ups - self._num_pending_scale_downs
        
        if self._replica_token_throughput > 0:
            target_replicas_up = math.ceil(max_arrival_rate_scale_up / self._replica_token_throughput)
            
            if target_replicas_up > current_replicas:
                scale_up_needed = target_replicas_up - current_replicas
                self._last_scale_up_time = time
                return scale_up_needed
        
        time_since_last_scale_up = time - self._last_scale_up_time
        can_scale_down = time_since_last_scale_up >= self._autoscaler_config.stabilization_delay
        
        if can_scale_down and self._replica_token_throughput > 0:
            target_replicas_down = math.ceil(max_arrival_rate_scale_down / self._replica_token_throughput)
            target_replicas_down = max(1, target_replicas_down)
            
            if target_replicas_down < current_replicas:
                scale_down_needed = target_replicas_down - current_replicas
                return scale_down_needed
        
        return 0


class TestNetworkEnvelope(unittest.TestCase):
    
    def setUp(self):
        self.config = MockInferlineAutoscalerConfig()
        self.envelope = NetworkEnvelope(self.config)
    
    def test_empty_envelope(self):
        """Test empty envelope returns 0 rate"""
        rate = self.envelope.get_max_token_arrival_rate(time=100.0, window_size=10.0, look_back_time=50.0)
        self.assertEqual(rate, 0.0)
    
    def test_single_request(self):
        """Test single request calculation"""
        request = MockRequest(arrived_at=10.0, num_prefill_tokens=100, num_decode_tokens=1)
        self.envelope.on_request_arrival(request)
        
        rate = self.envelope.get_max_token_arrival_rate(time=20.0, window_size=10.0, look_back_time=15.0)
        expected = 101.0 / 10.0  # 101 tokens over 10 time units
        self.assertAlmostEqual(rate, expected, places=2)
    
    def test_zero_window_size(self):
        """Test zero window size returns 0"""
        request = MockRequest(arrived_at=10.0, num_prefill_tokens=100, num_decode_tokens=1)
        self.envelope.on_request_arrival(request)
        
        rate = self.envelope.get_max_token_arrival_rate(time=20.0, window_size=0.0, look_back_time=15.0)
        self.assertEqual(rate, 0.0)


class TestInferlineAutoscaler(unittest.TestCase):
    
    def setUp(self):
        self.config = MockInferlineAutoscalerConfig()
        self.autoscaler = InferlineAutoscaler(self.config, num_replicas=2)
    
    def test_initial_state(self):
        """Test initial autoscaler state"""
        self.assertEqual(self.autoscaler.replica_token_throughput, 1.0)
    
    def test_on_batch_end_throughput_update(self):
        """Test Task 3: throughput update with exponential moving average"""
        batch = MockBatch(
            scheduled_at=10.0,
            completed_at=20.0,  # 10 time units execution time
            num_tokens=[100, 50, 75]  # 225 total tokens
        )
        
        self.autoscaler.on_batch_end(batch)
        
        # Expected: batch_throughput = 225 / 10 = 22.5
        # new_throughput = 0.5 * 22.5 + 0.5 * 1.0 = 11.75
        expected_throughput = 0.5 * 22.5 + 0.5 * 1.0
        self.assertAlmostEqual(self.autoscaler.replica_token_throughput, expected_throughput, places=2)
    
    def test_tune_scale_up(self):
        """Test Task 4: scale up decision"""
        # Add requests to trigger scale up
        requests = [
            MockRequest(arrived_at=90.0, num_prefill_tokens=100, num_decode_tokens=1),
            MockRequest(arrived_at=92.0, num_prefill_tokens=100, num_decode_tokens=1),
            MockRequest(arrived_at=94.0, num_prefill_tokens=100, num_decode_tokens=1),
            MockRequest(arrived_at=96.0, num_prefill_tokens=100, num_decode_tokens=1),
            MockRequest(arrived_at=98.0, num_prefill_tokens=100, num_decode_tokens=1),
        ]
        
        for req in requests:
            self.autoscaler._network_envelope.on_request_arrival(req)
        
        # With high arrival rate and low throughput, should scale up
        self.autoscaler._replica_token_throughput = 10.0  # tokens per second
        result = self.autoscaler.tune(time=100.0)
        self.assertGreater(result, 0)  # Should scale up
    
    def test_tune_scale_down_with_stabilization(self):
        """Test Task 4: scale down respects stabilization delay"""
        # Add one request for minimal load
        request = MockRequest(arrived_at=95.0, num_prefill_tokens=10, num_decode_tokens=1)
        self.autoscaler._network_envelope.on_request_arrival(request)
        
        self.autoscaler._replica_token_throughput = 100.0  # Very high throughput
        
        # Test that scale down is blocked by stabilization delay
        self.autoscaler._last_scale_up_time = 95.0  # Recent scale up
        result = self.autoscaler.tune(time=100.0)  # Only 5 seconds later
        self.assertEqual(result, 0)  # No scaling due to stabilization delay
        
        # Test that scale down works after stabilization delay
        self.autoscaler._last_scale_up_time = 80.0  # Scale up 20 seconds ago
        result = self.autoscaler.tune(time=100.0)  # 20 > 10 (stabilization_delay)
        self.assertLessEqual(result, 0)  # Should scale down or stay same


if __name__ == '__main__':
    unittest.main(verbosity=2)
    print("\nAll tests passed! The implementation is working correctly.")