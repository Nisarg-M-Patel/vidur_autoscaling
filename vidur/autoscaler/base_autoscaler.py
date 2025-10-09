from abc import ABC, abstractmethod
from typing import final

from vidur.config.config import BaseAutoscalerConfig
from vidur.entities.batch import Batch
from vidur.entities.cluster import Cluster
from vidur.entities.request import Request
from vidur.logger import init_logger
from vidur.metrics.metrics_store import MetricsStore
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

logger = init_logger(__name__)

class BaseAutoscaler(ABC):
    def __init__(
        self,
        autoscaler_config: BaseAutoscalerConfig,
        cluster: Cluster,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
    ) -> None:
        self._autoscaler_config = autoscaler_config
        self._cluster = cluster
        self._scheduler = scheduler
        self._metrics_store = metrics_store
        self._num_pending_scale_ups = 0
        self._num_pending_scale_downs = 0

    @property
    def cost_per_hour(self) -> float:
        return self._cluster.cost_per_hour

    @property
    def num_replicas(self) -> int:
        return self._cluster.num_replicas

    @final
    def add_replica(self):
        replica = self._cluster.add_replica()
        self._scheduler.add_replica(replica)
        self._metrics_store.add_replica(replica.id)
        self._num_pending_scale_ups -= 1

    @final
    def free_replica(self) -> None:
        # If there is an empty replica, free it
        for replica_id in self._scheduler._replica_schedulers:
            replica_scheduler = self._scheduler.get_replica_scheduler(replica_id)
            if replica_scheduler.is_empty() and not self._scheduler.check_replica_to_free(replica_id):
                self.free_replica_with_id(replica_id)
                return
        
        # If there is no empty replica, mark a replica to free
        self._scheduler.mark_replica_to_free()

    @final
    def free_replica_with_id(self, replica_id: int) -> None:
        replica_scheduler = self._scheduler.get_replica_scheduler(replica_id)

        assert replica_scheduler.is_empty()

        self._cluster.free_replica_with_id(replica_id)
        self._scheduler.free_replica_with_id(replica_id)
        self._num_pending_scale_downs -= 1

    @abstractmethod
    def tune(self) -> int:
        pass
    
    @abstractmethod
    def on_request_arrival(self, request: Request) -> None:
        pass

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass