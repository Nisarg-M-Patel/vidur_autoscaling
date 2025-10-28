from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LORGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    def mark_replica_to_free(self) -> int | None:
        """
        Part 1, Task 2
        Mark the replica with the least number of outstanding requests to be freed.
        Return the replica id marked to be freed.
        HINTS:
        1. The mark_replica_to_free method is required to mark the replica to free (i.e add the identified replica to the replicas_to_free set)
           and also return the marked replica id as hinted through the function return signature.
        """
        min_outstanding_requests = float('inf')
        replica_to_free = None

        for replica_id, replica_scheduler in self._replica_schedulers.items():
            #skip if alr marked to be freed?
            if self.check_replica_to_free(replica_id):
                continue

            outstanding_requests = (
                len(replica_scheduler._request_queue) + 
                len(replica_scheduler._allocation_map)
            )

            if outstanding_requests < min_outstanding_requests:
                min_outstanding_requests = outstanding_requests
                replica_to_free = replica_id

        if replica_to_free is not None:
            self._replicas_to_free.add(replica_to_free)
            return replica_to_free
        
        return None

    def schedule(self) -> List[Tuple[int, Request]]:
        """
        Part 1, Task 2
        Route requests to replicas based on the number of outstanding requests.
        """
        self.sort_requests()

        request_mapping = []

        available_replica_ids = [
            replica_id for replica_id in self._replica_schedulers.keys()
            if not self.check_replica_to_free(replica_id)
        ]

        if not available_replica_ids:
            return request_mapping
        
        while self._request_queue:
            request = self._request_queue.pop(0)

            min_outstanding_requests = float('inf')
            selected_replica_id = None
            
            for replica_id in available_replica_ids:
                replica_scheduler = self._replica_schedulers[replica_id]
                
                # Calculate outstanding requests: pending in queue + currently being processed
                outstanding_requests = (
                    len(replica_scheduler._request_queue) + 
                    len(replica_scheduler._allocation_map)
                )
                
                if outstanding_requests < min_outstanding_requests:
                    min_outstanding_requests = outstanding_requests
                    selected_replica_id = replica_id
            
            # Assign the request to the selected replica
            if selected_replica_id is not None:
                request_mapping.append((selected_replica_id, request))
        
        return request_mapping
            
        
