from typing import List

from vidur.autoscaler import BaseAutoscaler
from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
        autoscaler: BaseAutoscaler,
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        self._batch.on_batch_end(self.time)
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        if autoscaler is not None:
            autoscaler.on_batch_end(self._batch)

        next_events = []
        if not replica_scheduler.is_empty():
            next_events.append(ReplicaScheduleEvent(self.time, self._replica_id))
        else:
            if scheduler.check_replica_to_free(self._replica_id):
                autoscaler.free_replica_with_id(self._replica_id)
                metrics_store.on_autoscaling_event(
                    self.time, autoscaler.num_replicas, autoscaler.cost_per_hour
                )

        return next_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }
