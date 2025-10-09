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

        pass

    def on_request_arrival(self, request: Request) -> None:
        """
        Update the network envelope with the arrival of a new request.
        """
        pass

    def get_max_request_rate(self, time: float, window_size: float, look_back_time: float) -> float:
        """
        Args:
        - time: Current time
        - window_size: Window size for calculating the request token rates
        - look_back_time: Look back time for calculating the max request token rate
        
        Returns:
        - max_request_rate: Maximum request token rate in the window of size window_size in the duration between time - look_back_time and time
        """
        pass

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

    @property
    def replica_token_throughput(self) -> float:
        return self._replica_token_throughput

    def on_request_arrival(self, request: Request) -> None:
        """
        Update required state when a request arrives.
        """
        pass

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