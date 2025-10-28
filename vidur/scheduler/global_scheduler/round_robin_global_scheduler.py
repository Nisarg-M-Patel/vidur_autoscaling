from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class RoundRobinGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_counter = 0

    def schedule(self) -> List[Tuple[int, Request]]:
        """
        Part 1, Task 1
        Ensure that no new requests are routed to a replica that is marked to be freed.
        HINTS:
        1. Use self.check_replica_to_free(replica_id) to check if a replica is marked to be freed.
        2. Note that the provided RoundRobinGlobalScheduler is unaware of scale down actions and makes the assumption that replica ids range from 0 to num_replicas-1.
           However, this doesn't hold in the event of replica scale downs when some replicas get removed from the cluster.
           Therefore, make sure to use self._replicas or self._replica_schedulers to cycle through the replica ids.
        """
        self.sort_requests()

        request_mapping = []

        # get list of available replica IDs not marked to be freed

        available_replica_ids = sorted([
            replica_id for replica_id in self._replica_schedulers.keys()
            if not self.check_replica_to_free(replica_id)
        ])

        #if none available return empty list
        if not available_replica_ids:
            return request_mapping
        

        while self._request_queue:
            request = self._request_queue.pop(0)
            #replica_id = self._request_counter % self._num_replicas
            replica_idx = self._request_counter % len(available_replica_ids)
            replica_id = available_replica_ids[replica_idx]
            self._request_counter += 1
            request_mapping.append((replica_id, request))

        return request_mapping
