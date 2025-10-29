import math
from collections import deque
from vidur.autoscaler.base_autoscaler import BaseAutoscaler
from vidur.config.config import InferlineAutoscalerConfig
from vidur.entities.batch import Batch
from vidur.entities.cluster import Cluster
from vidur.entities.request import Request
from vidur.logger import init_logger
from vidur.metrics.metrics_store import MetricsStore
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

logger = init_logger(__name__)

"""
Network Traffic Envelope maintains a sliding window of request arrival token rates.
"""
class NetworkEnvelope:
    def __init__(self, autoscaler_config: InferlineAutoscalerConfig) -> None:
        """
        You will need the following fields from the Inferline Autoscaler config:
        - min_window_size_scale_up: Minimum window size for scale up
        - min_window_size_scale_down: Minimum window size for scale down
        - look_back_time_scale_up: Look back time for scale up
        - look_back_time_scale_down: Look back time for scale down
        """
        self._min_window_size_scale_up = autoscaler_config.min_window_size_scale_up
        self._min_window_size_scale_down = autoscaler_config.min_window_size_scale_down
        self._look_back_time_scale_up = autoscaler_config.look_back_time_scale_up
        self._look_back_time_scale_down = autoscaler_config.look_back_time_scale_down

        self._arrivals = deque()
        

    def on_request_arrival(self, request: Request) -> None:
        """
        Update the network envelope with the arrival of a new request.
        """
        total_tokens = request.num_prefill_tokens + request.num_decode_tokens
        self._arrivals.append((request.arrived_at, total_tokens))

    def get_max_request_rate(self, time: float, window_size: float, look_back_time: float) -> float:
        """
        Args:
        - time: Current time
        - window_size: Window size for calculating the request token rates
        - look_back_time: Look back time for calculating the max request token rate
        
        Returns:
        - max_request_rate: Maximum request token rate in the window of size window_size in the duration between time - look_back_time and time
        """
        if window_size <= 0:
            return 0.0
        
        # Clean old arrivals beyond lookback time
        cutoff_time = time - look_back_time
        while self._arrivals and self._arrivals[0][0] < cutoff_time:
            self._arrivals.popleft()
        
        if not self._arrivals:
            return 0.0
        
        # Much simpler approach: sample the window at regular intervals
        # This gives a good approximation while being O(1) per query
        max_rate = 0.0
        
        # Sample at most 10 window positions to keep it fast
        num_samples = min(10, int(look_back_time / window_size) + 1)
        
        if num_samples <= 1:
            # Single window case
            window_start = max(cutoff_time, time - window_size)
            window_end = time
            tokens_in_window = 0
            for arrival_time, tokens in self._arrivals:
                if window_start <= arrival_time < window_end:
                    tokens_in_window += tokens
            if window_end > window_start:
                return tokens_in_window / (window_end - window_start)
            return 0.0
        
        # Sample multiple positions
        step = (time - window_size - cutoff_time) / (num_samples - 1) if num_samples > 1 else 0
        
        for i in range(num_samples):
            window_start = cutoff_time + i * step
            window_end = window_start + window_size
            
            if window_end > time:
                window_end = time
                window_start = time - window_size
            
            tokens_in_window = 0
            for arrival_time, tokens in self._arrivals:
                if window_start <= arrival_time < window_end:
                    tokens_in_window += tokens
            
            rate = tokens_in_window / window_size
            max_rate = max(max_rate, rate)
        
        return max_rate


        


    

class InferlineAutoscaler(BaseAutoscaler):
    def __init__(
        self,
        autoscaler_config: InferlineAutoscalerConfig,
        cluster: Cluster,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
    ) -> None:
        super().__init__(autoscaler_config, cluster, scheduler, metrics_store)

        """
        You will need the following fields from the Inferline Autoscaler config for maintaining and updating the replica token throughput:
        - initial_replica_token_throughput: Initial replica token throughput
        - throughput_alpha: Alpha value for exponential moving average of replica token throughput
        """    
    
        self._replica_token_throughput = self._autoscaler_config.initial_replica_token_throughput
        self._throughput_alpha = self._autoscaler_config.throughput_alpha
        self._last_scale_up_time = -float('inf')

        self._network_envelope = NetworkEnvelope(autoscaler_config)


    @property
    def replica_token_throughput(self) -> float:
        return self._replica_token_throughput

    def on_request_arrival(self, request: Request) -> None:
        """
        Update required state when a request arrives.
        """
        self._network_envelope.on_request_arrival(request)

    def on_batch_end(self, batch: Batch) -> None:
        """
        Update the replica token throughput based on completed batch. Use exponential moving average to update the replica token throughput.
        """
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
        """
        Inferline based autoscaler tuning.
        Args:
        - time: Current time
        Returns:
        - replicas: Number of replicas to scale up or down
            +ve value indicates scale up
            -ve value indicates scale down
            0 indicates no change

        HINTS:
        1. The autoscaler config field stabilization_delay is the time to wait before scaling down after the last scale up.
        2. Check for scale up first, then scale down.
        3. self._num_pending_scale_ups and self._num_pending_scale_downs are the number of pending scale ups and scale downs respectively. Make sure to account for these.
        """
        max_arrival_rate_scale_up = self._network_envelope.get_max_request_rate(
            time, 
            self._autoscaler_config.min_window_size_scale_up,
            self._autoscaler_config.look_back_time_scale_up
        )
        
        max_arrival_rate_scale_down = self._network_envelope.get_max_request_rate(
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