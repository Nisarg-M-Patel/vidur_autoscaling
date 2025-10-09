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
        pass

    def on_request_arrival(self, request: Request) -> None:
        """
        Update required state when a request arrives.
        """
        pass

    def on_batch_end(self, batch: Batch) -> None:
        """
        Update required state when a batch ends.
        """
        pass

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
        pass