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
        self._arrivals.append(request.arrived_at, total_tokens)

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
        
        #remove old reqs that are beyond lookback time
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
            
            step_size = window_size / 10.0 #potential step
            window_start += step_size
        
        return max_rate
        

##optimized network envelope##
from collections import defaultdict, deque
import bisect

class OptimizedNetworkEnvelope:
    def __init__(self, autoscaler_config, bucket_size=1.0):
        self._min_window_size_scale_up = autoscaler_config.min_window_size_scale_up
        self._min_window_size_scale_down = autoscaler_config.min_window_size_scale_down
        self._look_back_time_scale_up = autoscaler_config.look_back_time_scale_up
        self._look_back_time_scale_down = autoscaler_config.look_back_time_scale_down
        
        self._bucket_size = bucket_size
        self._buckets = defaultdict(int) 
        self._bucket_times = []  
        
    def _get_bucket_id(self, time):
        return int(time // self._bucket_size)
    
    def _get_bucket_start(self, bucket_id):
        return bucket_id * self._bucket_size

    def on_request_arrival(self, request):
        total_tokens = request.num_prefill_tokens + request.num_decode_tokens
        bucket_id = self._get_bucket_id(request.arrived_at)
        
        if bucket_id not in self._buckets:
            bucket_start = self._get_bucket_start(bucket_id)
            bisect.insort(self._bucket_times, bucket_start)
        
        self._buckets[bucket_id] += total_tokens

    def get_max_request_rate(self, time, window_size, look_back_time):
        if window_size <= 0:
            return 0.0
            
        cutoff_time = time - look_back_time
        while self._bucket_times and self._bucket_times[0] < cutoff_time:
            old_bucket_start = self._bucket_times.pop(0)
            old_bucket_id = self._get_bucket_id(old_bucket_start)
            del self._buckets[old_bucket_id]
        
        if not self._buckets:
            return 0.0
        
        
        
        max_rate = 0.0
        num_buckets_in_window = max(1, int(window_size / self._bucket_size))
        
        start_bucket_id = self._get_bucket_id(cutoff_time)
        end_bucket_id = self._get_bucket_id(time)
        
        for bucket_start in range(start_bucket_id, end_bucket_id - num_buckets_in_window + 2):
            window_tokens = 0
            
            for i in range(num_buckets_in_window):
                bucket_id = bucket_start + i
                if bucket_id in self._buckets:
                    window_tokens += self._buckets[bucket_id]
            
            actual_window_size = min(window_size, num_buckets_in_window * self._bucket_size)
            if actual_window_size > 0:
                rate = window_tokens / actual_window_size
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
        pass
    
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
        pass