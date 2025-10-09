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
        pass

    def schedule(self) -> List[Tuple[int, Request]]:
        """
        Part 1, Task 2
        Route requests to replicas based on the number of outstanding requests.
        """
        pass
