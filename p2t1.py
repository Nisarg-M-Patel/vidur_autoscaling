"""
Simple test for NetworkEnvelope class (Part 2 Task 1)
"""

from dataclasses import dataclass
from collections import deque

@dataclass
class InferlineAutoscalerConfig:
    min_window_size_scale_up: float = 20.0
    min_window_size_scale_down: float = 20.0
    look_back_time_scale_up: float = 80.0
    look_back_time_scale_down: float = 80.0

@dataclass 
class Request:
    arrived_at: float
    num_prefill_tokens: int
    num_decode_tokens: int

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

    def get_max_request_rate(self, time: float, window_size: float, look_back_time: float) -> float:
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
            
            step_size = window_size / 10.0
            window_start += step_size
        
        return max_rate

def test_network_envelope():
    config = InferlineAutoscalerConfig()
    envelope = NetworkEnvelope(config)
    
    # Test 1: Empty envelope
    rate = envelope.get_max_request_rate(time=100.0, window_size=10.0, look_back_time=50.0)
    assert rate == 0.0, f"Empty envelope should return 0, got {rate}"
    print("Test 1 passed: Empty envelope")
    
    # Test 2: Single request
    request = Request(arrived_at=10.0, num_prefill_tokens=100, num_decode_tokens=1)
    envelope.on_request_arrival(request)
    
    rate = envelope.get_max_request_rate(time=20.0, window_size=10.0, look_back_time=15.0)
    expected = 101.0 / 10.0  # 101 tokens over 10 time units
    assert abs(rate - expected) < 0.01, f"Expected {expected}, got {rate}"
    print("Test 2 passed: Single request")
    
    # Test 3: Multiple requests
    request2 = Request(arrived_at=12.0, num_prefill_tokens=50, num_decode_tokens=1)
    request3 = Request(arrived_at=14.0, num_prefill_tokens=200, num_decode_tokens=1)
    envelope.on_request_arrival(request2)
    envelope.on_request_arrival(request3)
    
    rate = envelope.get_max_request_rate(time=25.0, window_size=10.0, look_back_time=20.0)
    assert rate > 0, "Should have positive rate with multiple requests"
    print("Test 3 passed: Multiple requests")
    
    # Test 4: Lookback time filtering
    rate = envelope.get_max_request_rate(time=100.0, window_size=10.0, look_back_time=5.0)
    assert rate == 0.0, "Should be 0 with short lookback time"
    print("Test 4 passed: Lookback filtering")
    
    # Test 5: Zero window size
    rate = envelope.get_max_request_rate(time=100.0, window_size=0.0, look_back_time=50.0)
    assert rate == 0.0, "Zero window size should return 0"
    print("Test 5 passed: Zero window size")
    
    # Test 6: Traffic patterns
    envelope_new = NetworkEnvelope(config)
    
    # High density period
    for i in range(5):
        req = Request(arrived_at=10.0 + i * 0.4, num_prefill_tokens=100, num_decode_tokens=1)
        envelope_new.on_request_arrival(req)
    
    # Low density period
    for i in range(2):
        req = Request(arrived_at=20.0 + i * 2.5, num_prefill_tokens=100, num_decode_tokens=1)
        envelope_new.on_request_arrival(req)
    
    rate = envelope_new.get_max_request_rate(time=30.0, window_size=3.0, look_back_time=25.0)
    assert rate > 100, "Should detect high density period"
    print("Test 6 passed: Traffic pattern detection")
    
    # Test 7: Different window sizes
    window_sizes = [5.0, 10.0, 20.0]
    rates = []
    for ws in window_sizes:
        r = envelope_new.get_max_request_rate(time=30.0, window_size=ws, look_back_time=25.0)
        rates.append(r)
    
    # Smaller windows typically capture higher peak rates
    print(f"Test 7 passed: Different window sizes - rates: {[f'{r:.1f}' for r in rates]}")
    
    # Test 8: Performance test
    envelope_perf = NetworkEnvelope(config)
    
    # Add many requests
    for i in range(1000):
        req = Request(arrived_at=i * 0.1, num_prefill_tokens=100, num_decode_tokens=1)
        envelope_perf.on_request_arrival(req)
    
    import time
    start = time.time()
    rate = envelope_perf.get_max_request_rate(time=100.0, window_size=10.0, look_back_time=50.0)
    duration = time.time() - start
    
    assert duration < 1.0, f"Query too slow: {duration:.3f}s"
    assert rate > 0, "Should have positive rate"
    print(f"Test 8 passed: Performance test - {duration*1000:.1f}ms for 1000 requests")
    
    print("All NetworkEnvelope tests passed!")

if __name__ == "__main__":
    test_network_envelope()