import unittest
from vidur.config.config import SimulationConfig, ClusterConfig
from vidur.entities.cluster import Cluster
from vidur.entities.request import Request
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import RoundRobinGlobalScheduler


class TestRoundRobinGlobalSchedulerScaleDown(unittest.TestCase):
    """
    Unit test for Part 1, Task 1: Round Robin Scheduler with Scale Down
    
    Test Case: Given a system with 3 replicas where one replica is marked for
    scaling down, verify that 5 incoming requests are distributed only among
    the remaining 2 replicas in round-robin fashion.
    
    The replica marked for scale down should receive no requests.
    """
    
    def setUp(self):
        """Initialize the simulation components"""
        # Create config with 3 replicas
        cluster_config = ClusterConfig(num_replicas=3)
        self.simulation_config = SimulationConfig(cluster_config=cluster_config)
        
        # Initialize cluster with replicas
        self.cluster = Cluster(
            cluster_config=self.simulation_config.cluster_config,
            metrics_config=self.simulation_config.metrics_config,
            generator_config=self.simulation_config.request_generator_config,
        )
        
        # Initialize Round Robin Scheduler
        self.scheduler = RoundRobinGlobalScheduler(
            self.simulation_config,
            self.cluster.replicas,
        )
        
        # Get the actual replica IDs (they might not be 0, 1, 2)
        self.replica_ids = sorted(list(self.cluster.replicas.keys()))
    
    def test_scale_down_avoidance(self):
        """
        Test that scheduler avoids routing to replicas marked for scaling down
        """
        # Verify we have 3 replicas
        self.assertEqual(len(self.replica_ids), 3, 
                        "Should have 3 replicas for this test")
        
        # Mark the middle replica for scaling down
        replica_to_free = self.replica_ids[1]
        self.scheduler._replicas_to_free.add(replica_to_free)
        
        # The remaining replicas should receive requests
        remaining_replicas = [self.replica_ids[0], self.replica_ids[2]]
        
        # Create 5 test requests
        requests = []
        for i in range(5):
            request = Request(
                arrived_at=float(i),
                num_prefill_tokens=10,
                num_decode_tokens=20
            )
            requests.append(request)
            self.scheduler.add_request(request)
        
        # Schedule the requests
        request_mapping = self.scheduler.schedule()
        
        # Verify we got 5 mappings
        self.assertEqual(len(request_mapping), 5, 
                        "Should have 5 request mappings")
        
        # Extract replica IDs that received requests
        assigned_replica_ids = [replica_id for replica_id, _ in request_mapping]
        
        # Verify no requests went to the replica marked for scale down
        self.assertNotIn(replica_to_free, assigned_replica_ids,
                        f"Replica {replica_to_free} should not receive any requests (marked for scale down)")
        
        # Verify requests are distributed between the two remaining replicas in round-robin
        expected_distribution = [
            remaining_replicas[0],  # Request 0
            remaining_replicas[1],  # Request 1
            remaining_replicas[0],  # Request 2
            remaining_replicas[1],  # Request 3
            remaining_replicas[0],  # Request 4
        ]
        self.assertEqual(assigned_replica_ids, expected_distribution,
                        f"Requests should be distributed as {expected_distribution} "
                        f"but got {assigned_replica_ids}")
        
        # Verify all requests were assigned to valid remaining replicas
        for replica_id in assigned_replica_ids:
            self.assertIn(replica_id, remaining_replicas,
                         f"All requests should go to replicas {remaining_replicas}, found {replica_id}")
        
        print(f"\n✓ Test passed! Replicas {remaining_replicas} received requests in round-robin fashion.")
        print(f"✓ Replica {replica_to_free} (marked for scale down) correctly received no requests.")
    
    def test_no_available_replicas(self):
        """
        Test that scheduler returns empty list when all replicas are marked for scale down
        """
        # Mark all replicas for scaling down
        for replica_id in self.replica_ids:
            self.scheduler._replicas_to_free.add(replica_id)
        
        # Create test requests
        for i in range(3):
            request = Request(
                arrived_at=float(i),
                num_prefill_tokens=10,
                num_decode_tokens=20
            )
            self.scheduler.add_request(request)
        
        # Schedule the requests
        request_mapping = self.scheduler.schedule()
        
        # Verify no requests were scheduled
        self.assertEqual(len(request_mapping), 0,
                        "Should return empty list when no replicas available")
        
        print(f"\n✓ Test passed! Scheduler correctly returns empty list when all replicas marked for scale down.")


if __name__ == '__main__':
    unittest.main()