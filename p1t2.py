"""
Unit tests for LORGlobalScheduler
Run with: python -m pytest test_lor_scheduler.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock
from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from vidur.entities import Request


class TestLORGlobalScheduler:
    
    def create_mock_replica_scheduler(self, replica_id, num_pending, num_allocated):
        """Helper to create a mock replica scheduler"""
        mock_scheduler = Mock()
        mock_scheduler.replica_id = replica_id
        mock_scheduler._request_queue = [Mock() for _ in range(num_pending)]
        mock_scheduler._allocation_map = {i: 1 for i in range(num_allocated)}
        return mock_scheduler
    
    def create_mock_scheduler(self):
        """Create a mock LOR scheduler with mock replicas"""
        scheduler = Mock(spec=LORGlobalScheduler)
        scheduler._replica_schedulers = {}
        scheduler._replicas_to_free = set()
        scheduler._request_queue = []
        
        # Bind the actual methods
        scheduler.check_replica_to_free = lambda rid: rid in scheduler._replicas_to_free
        scheduler.sort_requests = lambda: scheduler._request_queue.sort(key=lambda r: r.arrived_at)
        
        # Import the actual methods from LORGlobalScheduler
        scheduler.mark_replica_to_free = LORGlobalScheduler.mark_replica_to_free.__get__(scheduler)
        scheduler.schedule = LORGlobalScheduler.schedule.__get__(scheduler)
        
        return scheduler
    
    def test_mark_replica_to_free_selects_least_outstanding(self):
        """Test that mark_replica_to_free selects the replica with least outstanding requests"""
        scheduler = self.create_mock_scheduler()
        
        # Create replicas with different outstanding requests
        # Replica 0: 5 pending + 3 allocated = 8 total
        # Replica 1: 2 pending + 1 allocated = 3 total (should be selected)
        # Replica 2: 4 pending + 2 allocated = 6 total
        scheduler._replica_schedulers[0] = self.create_mock_replica_scheduler(0, 5, 3)
        scheduler._replica_schedulers[1] = self.create_mock_replica_scheduler(1, 2, 1)
        scheduler._replica_schedulers[2] = self.create_mock_replica_scheduler(2, 4, 2)
        
        result = scheduler.mark_replica_to_free()
        
        assert result == 1, "Should select replica 1 with least outstanding requests"
        assert 1 in scheduler._replicas_to_free, "Replica 1 should be marked to free"
    
    def test_mark_replica_to_free_skips_already_marked(self):
        """Test that mark_replica_to_free skips replicas already marked"""
        scheduler = self.create_mock_scheduler()
        
        # Create replicas
        scheduler._replica_schedulers[0] = self.create_mock_replica_scheduler(0, 2, 1)
        scheduler._replica_schedulers[1] = self.create_mock_replica_scheduler(1, 5, 2)
        
        # Mark replica 0 to be freed
        scheduler._replicas_to_free.add(0)
        
        result = scheduler.mark_replica_to_free()
        
        assert result == 1, "Should select replica 1 since replica 0 is already marked"
        assert 1 in scheduler._replicas_to_free, "Replica 1 should be marked to free"
    
    def test_mark_replica_to_free_returns_none_when_all_marked(self):
        """Test that mark_replica_to_free returns None when all replicas are marked"""
        scheduler = self.create_mock_scheduler()
        
        scheduler._replica_schedulers[0] = self.create_mock_replica_scheduler(0, 2, 1)
        scheduler._replica_schedulers[1] = self.create_mock_replica_scheduler(1, 3, 2)
        
        # Mark all replicas to be freed
        scheduler._replicas_to_free.add(0)
        scheduler._replicas_to_free.add(1)
        
        result = scheduler.mark_replica_to_free()
        
        assert result is None, "Should return None when all replicas are marked"
    
    def test_schedule_assigns_to_least_outstanding(self):
        """Test that schedule assigns requests to replica with least outstanding requests"""
        scheduler = self.create_mock_scheduler()
        
        # Create replicas with different outstanding requests
        scheduler._replica_schedulers[0] = self.create_mock_replica_scheduler(0, 5, 3)  # 8 total
        scheduler._replica_schedulers[1] = self.create_mock_replica_scheduler(1, 1, 1)  # 2 total
        scheduler._replica_schedulers[2] = self.create_mock_replica_scheduler(2, 3, 2)  # 5 total
        
        # Create mock requests
        req1 = Mock(spec=Request)
        req1.arrived_at = 1.0
        req2 = Mock(spec=Request)
        req2.arrived_at = 2.0
        
        scheduler._request_queue = [req1, req2]
        
        result = scheduler.schedule()
        
        # Both requests should be assigned to replica 1 (least outstanding)
        assert len(result) == 2, "Should have 2 request mappings"
        assert result[0] == (1, req1), "First request should go to replica 1"
        assert result[1] == (1, req2), "Second request should go to replica 1"
        assert len(scheduler._request_queue) == 0, "Request queue should be empty"
    
    def test_schedule_skips_replicas_to_free(self):
        """Test that schedule does not assign requests to replicas marked to be freed"""
        scheduler = self.create_mock_scheduler()
        
        scheduler._replica_schedulers[0] = self.create_mock_replica_scheduler(0, 0, 0)  # 0 total (marked)
        scheduler._replica_schedulers[1] = self.create_mock_replica_scheduler(1, 5, 3)  # 8 total
        
        # Mark replica 0 to be freed
        scheduler._replicas_to_free.add(0)
        
        req1 = Mock(spec=Request)
        req1.arrived_at = 1.0
        
        scheduler._request_queue = [req1]
        
        result = scheduler.schedule()
        
        # Request should go to replica 1, not replica 0
        assert len(result) == 1, "Should have 1 request mapping"
        assert result[0] == (1, req1), "Request should go to replica 1, not replica 0"
    
    def test_schedule_returns_empty_when_no_available_replicas(self):
        """Test that schedule returns empty list when no replicas are available"""
        scheduler = self.create_mock_scheduler()
        
        scheduler._replica_schedulers[0] = self.create_mock_replica_scheduler(0, 2, 1)
        scheduler._replicas_to_free.add(0)
        
        req1 = Mock(spec=Request)
        req1.arrived_at = 1.0
        
        scheduler._request_queue = [req1]
        
        result = scheduler.schedule()
        
        assert len(result) == 0, "Should return empty list when no replicas available"
        assert len(scheduler._request_queue) == 1, "Request should remain in queue when no replicas available"
    
    def test_schedule_balances_load_dynamically(self):
        """Test that schedule assigns requests based on current outstanding requests"""
        scheduler = self.create_mock_scheduler()
        
        # Create 3 replicas with different initial loads
        # Replica 0: 3 outstanding
        # Replica 1: 1 outstanding  (should get next requests)
        # Replica 2: 5 outstanding
        scheduler._replica_schedulers[0] = self.create_mock_replica_scheduler(0, 2, 1)  # 3 total
        scheduler._replica_schedulers[1] = self.create_mock_replica_scheduler(1, 0, 1)  # 1 total
        scheduler._replica_schedulers[2] = self.create_mock_replica_scheduler(2, 3, 2)  # 5 total
        
        # Create 3 requests
        requests = [Mock(spec=Request, arrived_at=float(i)) for i in range(3)]
        scheduler._request_queue = requests.copy()
        
        result = scheduler.schedule()
        
        assert len(result) == 3, "Should have 3 request mappings"
        
        # All requests should go to replica 1 (has the least outstanding requests)
        for replica_id, request in result:
            assert replica_id == 1, f"All requests should go to replica 1 (least outstanding), but got replica {replica_id}"


if __name__ == "__main__":
    # Run tests
    test = TestLORGlobalScheduler()
    
    print("Running test_mark_replica_to_free_selects_least_outstanding...")
    test.test_mark_replica_to_free_selects_least_outstanding()
    print("✓ Passed\n")
    
    print("Running test_mark_replica_to_free_skips_already_marked...")
    test.test_mark_replica_to_free_skips_already_marked()
    print("✓ Passed\n")
    
    print("Running test_mark_replica_to_free_returns_none_when_all_marked...")
    test.test_mark_replica_to_free_returns_none_when_all_marked()
    print("✓ Passed\n")
    
    print("Running test_schedule_assigns_to_least_outstanding...")
    test.test_schedule_assigns_to_least_outstanding()
    print("✓ Passed\n")
    
    print("Running test_schedule_skips_replicas_to_free...")
    test.test_schedule_skips_replicas_to_free()
    print("✓ Passed\n")
    
    print("Running test_schedule_returns_empty_when_no_available_replicas...")
    test.test_schedule_returns_empty_when_no_available_replicas()
    print("✓ Passed\n")
    
    print("Running test_schedule_balances_load_dynamically...")
    test.test_schedule_balances_load_dynamically()
    print("✓ Passed\n")
    
    print("All tests passed! ✓")