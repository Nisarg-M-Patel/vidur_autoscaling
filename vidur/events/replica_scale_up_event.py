from typing import List

from vidur.autoscaler import BaseAutoscaler
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaScaleUpEvent(BaseEvent):
    def __init__(self, time: float):
        super().__init__(time, EventType.REPLICA_SCALE_UP)

    def handle_event(
        self,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
        autoscaler: BaseAutoscaler,
    ) -> List[BaseEvent]:
        autoscaler.add_replica()
        metrics_store.on_autoscaling_event(
            self.time, autoscaler.num_replicas, autoscaler.cost_per_hour
        )
        return []
