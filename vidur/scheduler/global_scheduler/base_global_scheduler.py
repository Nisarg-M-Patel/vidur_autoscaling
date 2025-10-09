from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)


class BaseGlobalScheduler(ABC):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas
        self._replicas_to_free = set()

        self._num_replicas = len(self._replicas)

        self._execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )
        self._replica_schedulers = {
            replica_id: ReplicaSchedulerRegistry.get(
                config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=config.cluster_config.replica_config,
                replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,
                execution_time_predictor=self._execution_time_predictor,
            )
            for replica_id, replica in replicas.items()
        }
        self._request_queue = []

    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda request: request._arrived_at)

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_scheduler(self, replica_id: int):
        return self._replica_schedulers[replica_id]

    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int):
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    def add_replica(self, replica: Replica) -> None:
        self._replica_schedulers[replica.id] = ReplicaSchedulerRegistry.get(
            self._config.cluster_config.replica_scheduler_config.get_type(),
            replica_config=self._config.cluster_config.replica_config,
            replica_scheduler_config=self._config.cluster_config.replica_scheduler_config,
            request_generator_config=self._config.request_generator_config,
            replica=replica,
            num_stages=replica.num_pipeline_stages,
            execution_time_predictor=self._execution_time_predictor,
        )
        self._num_replicas += 1

    def free_replica_with_id(self, replica_id: int) -> None:
        if replica_id in self._replicas_to_free:
            self._replicas_to_free.remove(replica_id)

        del self._replica_schedulers[replica_id]
        self._num_replicas -= 1

    def replica_exists(self, replica_id: int) -> bool:
        return replica_id in self._replicas

    def check_replica_to_free(self, replica_id: int) -> bool:
        return replica_id in self._replicas_to_free

    def mark_replica_to_free(self) -> int | None:
        for replica_id in self._replica_schedulers:
            if not self.check_replica_to_free(replica_id):
                self._replicas_to_free.add(replica_id)
                return replica_id
        return None

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
