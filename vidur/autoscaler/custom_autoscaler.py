import math
from collections import deque
from vidur.autoscaler.base_autoscaler import BaseAutoscaler
from vidur.config.config import CustomAutoscalerConfig
from vidur.entities.batch import Batch
from vidur.entities.cluster import Cluster
from vidur.entities.request import Request
from vidur.logger import init_logger
from vidur.metrics.metrics_store import MetricsStore
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

logger = init_logger(__name__)

class NetworkEnvelope:
    """Custom Network Traffic Envelope for the CustomAutoscaler"""
    def __init__(self, min_window_size_scale_up: float, min_window_size_scale_down: float,
                 look_back_time_scale_up: float, look_back_time_scale_down: float) -> None:
        self._min_window_size_scale_up = min_window_size_scale_up
        self._min_window_size_scale_down = min_window_size_scale_down
        self._look_back_time_scale_up = look_back_time_scale_up
        self._look_back_time_scale_down = look_back_time_scale_down
        self._arrivals = deque()

    def on_request_arrival(self, request: Request) -> None:
        """Update the network envelope with the arrival of a new request."""
        total_tokens = request.num_prefill_tokens + request.num_decode_tokens
        self._arrivals.append((request.arrived_at, total_tokens))

    def get_max_request_rate(self, time: float, window_size: float, look_back_time: float) -> float:
        """Calculate maximum request token rate in the specified window and lookback time."""
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
    
class CustomAutoscaler(BaseAutoscaler):
    def __init__(
        self,
        autoscaler_config: CustomAutoscalerConfig,
        cluster: Cluster,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
    ) -> None:
        super().__init__(autoscaler_config, cluster, scheduler, metrics_store)
        self.init_service_level()
        """
        Initialize the autoscaler
        """



    def init_service_level(self) -> None:
        """
        Initialize the service level of the autoscaler
        The autoscaler config contains the field service_level that can be set to 1, 2, or 3
        Set up the required state based on the service level
        
        HINT: As an example, when using the InferlineAutoscaler, you can play around with min window sizes and look back times for different service levels
        """
        
        service_level = self._autoscaler_config.service_level
        
        if service_level == 1:
            self._look_back_time_scale_up = 70.0
            self._look_back_time_scale_down = 50.0  
            self._min_window_size_scale_up = 35.0
            self._min_window_size_scale_down = 25.0
            self._stabilization_delay = 15.0  
            self._initial_replica_token_throughput = 20.0
            self._throughput_alpha = 0.2  
            self._scale_up_threshold = 1.05  
            self._scale_down_threshold = 0.85  
            self._min_replicas = 1
            self._max_scale_per_decision = 2
            
        elif service_level == 2:
            self._look_back_time_scale_up = 70.0
            self._look_back_time_scale_down = 80.0
            self._min_window_size_scale_up = 30.0
            self._min_window_size_scale_down = 35.0
            self._stabilization_delay = 30.0
            self._initial_replica_token_throughput = 20.0
            self._throughput_alpha = 0.3
            self._scale_up_threshold = 1.1
            self._scale_down_threshold = 0.7
            self._min_replicas = 1
            self._max_scale_per_decision = 2
            
        elif service_level == 3:
            self._look_back_time_scale_up = 50.0 
            self._look_back_time_scale_down = 120.0  
            self._min_window_size_scale_up = 20.0
            self._min_window_size_scale_down = 45.0
            self._stabilization_delay = 50.0  
            self._initial_replica_token_throughput = 20.0
            self._throughput_alpha = 0.4 
            self._scale_up_threshold = 0.8  
            self._scale_down_threshold = 0.55  
            self._min_replicas = 1
            self._max_scale_per_decision = 3  
            
        else:
            service_level = 2
            self._look_back_time_scale_up = 70.0
            self._look_back_time_scale_down = 80.0
            self._min_window_size_scale_up = 30.0
            self._min_window_size_scale_down = 35.0
            self._stabilization_delay = 30.0
            self._initial_replica_token_throughput = 20.0
            self._throughput_alpha = 0.3
            self._scale_up_threshold = 1.1
            self._scale_down_threshold = 0.7
            self._min_replicas = 1
            self._max_scale_per_decision = 2
            
        self._replica_token_throughput = self._initial_replica_token_throughput
        self._last_scale_up_time = -float('inf')
        self._last_scale_down_time = -float('inf')
        
        self._network_envelope = NetworkEnvelope(
            self._min_window_size_scale_up,
            self._min_window_size_scale_down,
            self._look_back_time_scale_up,
            self._look_back_time_scale_down
        )

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
        Update required state when a batch ends.
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
        Args:
        - time: Current time
        Returns:
        - replicas: Number of replicas to scale up or down
            +ve value indicates scale up
            -ve value indicates scale down
            0 indicates no change
        """
        max_arrival_rate_scale_up = self._network_envelope.get_max_request_rate(
            time, 
            self._min_window_size_scale_up,
            self._look_back_time_scale_up
        )
        
        max_arrival_rate_scale_down = self._network_envelope.get_max_request_rate(
            time,
            self._min_window_size_scale_down,
            self._look_back_time_scale_down
        )
        
        current_replicas = self.num_replicas + self._num_pending_scale_ups - self._num_pending_scale_downs
        
        if self._replica_token_throughput <= 0:
            return 0
            
        current_capacity = current_replicas * self._replica_token_throughput
        required_capacity_up = max_arrival_rate_scale_up
        required_capacity_down = max_arrival_rate_scale_down
        
        if required_capacity_up > current_capacity * self._scale_up_threshold:
            target_replicas = math.ceil(required_capacity_up / self._replica_token_throughput)
            scale_up_needed = min(target_replicas - current_replicas, self._max_scale_per_decision)
            
            if scale_up_needed > 0:
                self._last_scale_up_time = time
                return scale_up_needed
        
        time_since_last_scale_up = time - self._last_scale_up_time
        time_since_last_scale_down = time - self._last_scale_down_time
        
        can_scale_down = (time_since_last_scale_up >= self._stabilization_delay and 
                         time_since_last_scale_down >= self._stabilization_delay / 2)
        
        if can_scale_down and required_capacity_down < current_capacity * self._scale_down_threshold:
            target_replicas = max(
                math.ceil(required_capacity_down / self._replica_token_throughput),
                self._min_replicas
            )
            
            scale_down_possible = current_replicas - target_replicas
            scale_down_needed = min(scale_down_possible, self._max_scale_per_decision)
            
            if scale_down_needed > 0:
                self._last_scale_down_time = time
                return -scale_down_needed
        
        return 0