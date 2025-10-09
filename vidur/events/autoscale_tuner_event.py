from typing import List

from vidur.autoscaler import BaseAutoscaler
from vidur.events.base_event import BaseEvent
from vidur.events.replica_scale_down_event import ReplicaScaleDownEvent
from vidur.events.replica_scale_up_event import ReplicaScaleUpEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class AutoscaleTunerEvent(BaseEvent):
    def __init__(self, time: float):
        super().__init__(time, EventType.AUTOSCALE_TUNER)

    def handle_event(
        self,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
        autoscaler: BaseAutoscaler,
    ) -> List[BaseEvent]:
        num_replicas = autoscaler.tune(self._time)
        next_events = [
            AutoscaleTunerEvent(self.time + autoscaler._autoscaler_config.tune_interval)
        ]
        if num_replicas > 0:
            next_events.extend(
                [
                    ReplicaScaleUpEvent(
                        self.time + autoscaler._autoscaler_config.scale_up_delay
                    )
                    for _ in range(num_replicas)
                ]
            )
            autoscaler._num_pending_scale_ups += num_replicas
        elif num_replicas < 0:
            next_events.extend(
                [
                    ReplicaScaleDownEvent(
                        self.time + autoscaler._autoscaler_config.scale_down_delay
                    )
                    for _ in range(-num_replicas)
                ]
            )
            autoscaler._num_pending_scale_downs += abs(num_replicas)

        return next_events
