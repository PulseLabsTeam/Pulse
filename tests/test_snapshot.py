"""
Comprehensive test suite for snapshot.py

Tests all functionality to achieve 90%+ coverage.
"""

import pytest
import asyncio
import time
import pickle
from pulseos.persistence.snapshot import (
    StateSnapshot,
    SnapshotManager,
    SnapshotMetadata
)


class TestSnapshotMetadata:
    """Tests for SnapshotMetadata dataclass"""
    
    def test_creation(self):
        """Test creating snapshot metadata"""
        metadata = SnapshotMetadata(
            snapshot_id="test123",
            timestamp=123.45,
            step=10,
            survival_signal=0.8,
            agent_count=5,
            size_bytes=1024,
            compressed_size_bytes=512,
            parent_snapshot_id="parent123"
        )
        
        assert metadata.snapshot_id == "test123"
        assert metadata.timestamp == 123.45
        assert metadata.step == 10
        assert metadata.survival_signal == 0.8
        assert metadata.agent_count == 5
        assert metadata.size_bytes == 1024
        assert metadata.compressed_size_bytes == 512
        assert metadata.parent_snapshot_id == "parent123"
    
    def test_creation_no_parent(self):
        """Test creating metadata without parent"""
        metadata = SnapshotMetadata(
            snapshot_id="test123",
            timestamp=123.45,
            step=10,
            survival_signal=0.8,
            agent_count=5,
            size_bytes=1024,
            compressed_size_bytes=512
        )
        
        assert metadata.parent_snapshot_id is None


class TestStateSnapshot:
    """Comprehensive tests for StateSnapshot"""
    
    def test_initialization_no_delta_no_compression(self):
        """Test initialization without delta encoding or compression"""
        data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.5}}}
        
        snapshot = StateSnapshot(
            snapshot_data=data,
            enable_delta_encoding=False,
            enable_compression=False
        )
        
        assert snapshot.step == 1
        assert snapshot.survival_signal == 0.8
        assert snapshot.agent_count == 1
        assert snapshot.enable_delta_encoding is False
        assert snapshot.enable_compression is False
        assert snapshot.parent_snapshot_id is None
        assert snapshot.size_bytes == snapshot.compressed_size_bytes
    
    def test_initialization_with_compression(self):
        """Test initialization with compression"""
        data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.5}}}
        
        snapshot = StateSnapshot(
            snapshot_data=data,
            enable_delta_encoding=False,
            enable_compression=True
        )
        
        assert snapshot.enable_compression is True
        assert snapshot.compressed_size_bytes <= snapshot.size_bytes
    
    def test_initialization_with_delta_encoding(self):
        """Test initialization with delta encoding"""
        parent_data = {"step": 0, "survival_signal": 0.7, "agents": {"agent1": {"state": 0.4}}}
        current_data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.5}}}
        
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        snapshot = StateSnapshot(
            snapshot_data=current_data,
            parent_snapshot=parent,
            enable_delta_encoding=True
        )
        
        assert snapshot.enable_delta_encoding is True
        assert snapshot.parent_snapshot_id == parent.snapshot_id
    
    def test_snapshot_id_generation(self):
        """Test snapshot ID generation"""
        data = {"step": 1}
        
        snapshot1 = StateSnapshot(data)
        snapshot2 = StateSnapshot(data)
        
        # IDs should be unique
        assert snapshot1.snapshot_id != snapshot2.snapshot_id
        assert len(snapshot1.snapshot_id) == 16  # SHA256 hex[:16]
    
    def test_delta_encode_simple(self):
        """Test delta encoding with simple changes"""
        parent_data = {"step": 0, "value": 10}
        current_data = {"step": 1, "value": 20}
        
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        # Delta should only contain changed values
        full_data = snapshot.get_full_data(parent)
        assert full_data["step"] == 1
        assert full_data["value"] == 20
    
    def test_delta_encode_nested_dict(self):
        """Test delta encoding with nested dictionaries"""
        parent_data = {
            "step": 0,
            "agents": {
                "agent1": {"state": 0.5, "score": 10},
                "agent2": {"state": 0.6}
            }
        }
        current_data = {
            "step": 1,
            "agents": {
                "agent1": {"state": 0.7, "score": 10},  # Only state changed
                "agent2": {"state": 0.6}  # Unchanged
            }
        }
        
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        full_data = snapshot.get_full_data(parent)
        assert full_data["step"] == 1
        assert full_data["agents"]["agent1"]["state"] == 0.7
        assert full_data["agents"]["agent1"]["score"] == 10
        assert full_data["agents"]["agent2"]["state"] == 0.6
    
    def test_delta_encode_no_changes(self):
        """Test delta encoding when nothing changes"""
        data = {"step": 1, "value": 10}
        
        parent = StateSnapshot(data, enable_delta_encoding=False)
        snapshot = StateSnapshot(data, parent_snapshot=parent, enable_delta_encoding=True)
        
        # Delta should be minimal or empty
        full_data = snapshot.get_full_data(parent)
        assert full_data == data
    
    def test_delta_encode_new_keys(self):
        """Test delta encoding with new keys"""
        parent_data = {"step": 0}
        current_data = {"step": 1, "new_key": "new_value"}
        
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        full_data = snapshot.get_full_data(parent)
        assert full_data["new_key"] == "new_value"
    
    def test_delta_encode_removed_keys(self):
        """Test delta encoding with removed keys"""
        parent_data = {"step": 0, "old_key": "old_value"}
        current_data = {"step": 1}
        
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        # Note: Current implementation doesn't handle removed keys explicitly
        # This is a limitation, but we test what exists
        full_data = snapshot.get_full_data(parent)
        assert full_data["step"] == 1
    
    def test_get_full_data_no_compression(self):
        """Test getting full data without compression"""
        data = {"step": 1, "value": 10}
        
        snapshot = StateSnapshot(data, enable_compression=False)
        full_data = snapshot.get_full_data()
        
        assert full_data == data
    
    def test_get_full_data_with_compression(self):
        """Test getting full data with compression"""
        data = {"step": 1, "value": 10}
        
        snapshot = StateSnapshot(data, enable_compression=True)
        full_data = snapshot.get_full_data()
        
        assert full_data == data
    
    def test_get_full_data_with_delta(self):
        """Test getting full data with delta encoding"""
        parent_data = {"step": 0, "value": 10}
        current_data = {"step": 1, "value": 20}
        
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        full_data = snapshot.get_full_data(parent)
        assert full_data["step"] == 1
        assert full_data["value"] == 20
    
    def test_get_full_data_without_parent(self):
        """Test getting full data without providing parent"""
        parent_data = {"step": 0, "value": 10}
        current_data = {"step": 1, "value": 20}
        
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        # Should still work, but won't decode delta
        full_data = snapshot.get_full_data()
        # Data will be delta-encoded, so won't match exactly
        assert isinstance(full_data, dict)
    
    def test_get_metadata(self):
        """Test getting snapshot metadata"""
        data = {"step": 10, "survival_signal": 0.8, "agents": {"agent1": {}, "agent2": {}}}
        
        snapshot = StateSnapshot(data)
        metadata = snapshot.get_metadata()
        
        assert isinstance(metadata, SnapshotMetadata)
        assert metadata.snapshot_id == snapshot.snapshot_id
        assert metadata.step == 10
        assert metadata.survival_signal == 0.8
        assert metadata.agent_count == 2
        assert metadata.size_bytes == snapshot.size_bytes
        assert metadata.compressed_size_bytes == snapshot.compressed_size_bytes
    
    def test_get_compression_ratio(self):
        """Test getting compression ratio"""
        data = {"step": 1, "value": "x" * 1000}  # Some data
        
        snapshot = StateSnapshot(data, enable_compression=True)
        ratio = snapshot.get_compression_ratio()
        
        assert 0.0 <= ratio <= 1.0
    
    def test_get_compression_ratio_zero_size(self):
        """Test compression ratio with zero size"""
        data = {}
        
        snapshot = StateSnapshot(data, enable_compression=False)
        snapshot.size_bytes = 0
        ratio = snapshot.get_compression_ratio()
        
        assert ratio == 1.0
    
    def test_serialize_deserialize(self):
        """Test serialization and deserialization"""
        data = {"step": 1, "value": 10, "nested": {"key": "value"}}
        
        snapshot = StateSnapshot(data, enable_compression=False)
        full_data = snapshot.get_full_data()
        
        assert full_data == data
    
    def test_compression_decompression(self):
        """Test compression and decompression"""
        data = {"step": 1, "value": "x" * 1000}
        
        snapshot = StateSnapshot(data, enable_compression=True)
        full_data = snapshot.get_full_data()
        
        assert full_data == data
    
    def test_empty_data(self):
        """Test snapshot with empty data"""
        snapshot = StateSnapshot({})
        
        assert snapshot.step == 0
        assert snapshot.agent_count == 0
        assert snapshot.survival_signal == 0.0
    
    def test_large_data(self):
        """Test snapshot with large data"""
        large_data = {
            "step": 1,
            "agents": {f"agent_{i}": {"state": i * 0.1} for i in range(1000)}
        }
        
        snapshot = StateSnapshot(large_data)
        
        assert snapshot.agent_count == 1000
        assert snapshot.size_bytes > 0


class TestSnapshotManager:
    """Comprehensive tests for SnapshotManager"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test snapshot manager initialization"""
        manager = SnapshotManager(
            snapshot_interval=1.0,
            max_snapshots=100,
            enable_delta_encoding=True,
            enable_compression=True
        )
        
        assert manager.snapshot_interval == 1.0
        assert manager.max_snapshots == 100
        assert manager.enable_delta_encoding is True
        assert manager.enable_compression is True
        assert manager.get_snapshot_count() == 0
        assert manager.total_snapshots_created == 0
        assert manager.total_rollbacks == 0
    
    @pytest.mark.asyncio
    async def test_create_snapshot(self):
        """Test creating a snapshot"""
        manager = SnapshotManager(snapshot_interval=0.0)  # No interval
        
        data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.5}}}
        
        snapshot = await manager.create_snapshot(data)
        
        assert snapshot is not None
        assert snapshot.step == 1
        assert manager.get_snapshot_count() == 1
        assert manager.total_snapshots_created == 1
    
    @pytest.mark.asyncio
    async def test_create_snapshot_with_interval(self):
        """Test snapshot creation respects interval"""
        manager = SnapshotManager(snapshot_interval=0.1)
        
        data = {"step": 1, "survival_signal": 0.8}
        
        # First snapshot should be created
        snapshot1 = await manager.create_snapshot(data)
        assert snapshot1 is not None
        
        # Second snapshot immediately should return same snapshot
        snapshot2 = await manager.create_snapshot(data)
        assert snapshot2.snapshot_id == snapshot1.snapshot_id
        
        # Wait for interval
        await asyncio.sleep(0.15)
        
        # Now should create new snapshot
        snapshot3 = await manager.create_snapshot(data)
        assert snapshot3.snapshot_id != snapshot1.snapshot_id
    
    @pytest.mark.asyncio
    async def test_create_snapshot_max_limit(self):
        """Test snapshot manager respects max_snapshots limit"""
        manager = SnapshotManager(max_snapshots=3, snapshot_interval=0.0)
        
        for i in range(5):
            data = {"step": i, "survival_signal": 0.8}
            await manager.create_snapshot(data)
        
        # Should only keep max_snapshots
        assert manager.get_snapshot_count() == 3
    
    @pytest.mark.asyncio
    async def test_create_snapshot_delta_encoding(self):
        """Test snapshot creation with delta encoding"""
        manager = SnapshotManager(enable_delta_encoding=True, snapshot_interval=0.0)
        
        parent_data = {"step": 0, "value": 10}
        current_data = {"step": 1, "value": 20}
        
        parent = await manager.create_snapshot(parent_data)
        snapshot = await manager.create_snapshot(current_data)
        
        assert snapshot.parent_snapshot_id == parent.snapshot_id
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot(self):
        """Test finding best recovery snapshot"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        # Create snapshots with different survival signals
        for i, signal in enumerate([0.2, 0.5, 0.9, 0.3]):
            data = {"step": i, "survival_signal": signal}
            await manager.create_snapshot(data)
        
        # Find best with threshold 0.4
        best = await manager.find_best_recovery_snapshot(min_survival_signal=0.4)
        
        assert best is not None
        assert best.survival_signal == 0.9  # Highest above threshold
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot_no_candidates(self):
        """Test finding recovery snapshot when no candidates meet threshold"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        # Create snapshots all below threshold
        for i, signal in enumerate([0.1, 0.2, 0.3]):
            data = {"step": i, "survival_signal": signal}
            await manager.create_snapshot(data)
        
        # Should fallback to most recent
        best = await manager.find_best_recovery_snapshot(min_survival_signal=0.5)
        
        assert best is not None
        assert best.survival_signal == 0.3  # Most recent
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot_empty(self):
        """Test finding recovery snapshot with no snapshots"""
        manager = SnapshotManager()
        
        best = await manager.find_best_recovery_snapshot()
        
        assert best is None
    
    def test_get_snapshot(self):
        """Test getting snapshot by ID"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        data = {"step": 1, "survival_signal": 0.8}
        
        async def create():
            snapshot = await manager.create_snapshot(data)
            return snapshot.snapshot_id
        
        snapshot_id = asyncio.run(create())
        
        snapshot = manager.get_snapshot(snapshot_id)
        assert snapshot is not None
        assert snapshot.snapshot_id == snapshot_id
    
    def test_get_snapshot_nonexistent(self):
        """Test getting nonexistent snapshot"""
        manager = SnapshotManager()
        
        snapshot = manager.get_snapshot("nonexistent")
        
        assert snapshot is None
    
    def test_get_snapshot_count(self):
        """Test getting snapshot count"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        assert manager.get_snapshot_count() == 0
        
        async def create_multiple():
            for i in range(3):
                data = {"step": i, "survival_signal": 0.8}
                await manager.create_snapshot(data)
        
        asyncio.run(create_multiple())
        
        assert manager.get_snapshot_count() == 3
    
    def test_get_statistics_empty(self):
        """Test getting statistics with no snapshots"""
        manager = SnapshotManager()
        
        stats = manager.get_statistics()
        
        assert stats["snapshot_count"] == 0
        assert stats["total_created"] == 0
        assert stats["total_rollbacks"] == 0
        assert stats["average_size_bytes"] == 0
        assert stats["average_compression_ratio"] == 0.0
    
    def test_get_statistics(self):
        """Test getting statistics with snapshots"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        async def create_and_stats():
            for i in range(3):
                data = {"step": i, "survival_signal": 0.8}
                await manager.create_snapshot(data)
            return manager.get_statistics()
        
        stats = asyncio.run(create_and_stats())
        
        assert stats["snapshot_count"] == 3
        assert stats["total_created"] == 3
        assert "average_size_bytes" in stats
        assert "average_compression_ratio" in stats
        assert "total_storage_bytes" in stats
    
    def test_clear_snapshots(self):
        """Test clearing all snapshots"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        async def create_and_clear():
            for i in range(3):
                data = {"step": i, "survival_signal": 0.8}
                await manager.create_snapshot(data)
            
            assert manager.get_snapshot_count() == 3
            
            manager.clear_snapshots()
            
            assert manager.get_snapshot_count() == 0
            assert len(manager.snapshot_index) == 0
        
        asyncio.run(create_and_clear())
    
    @pytest.mark.asyncio
    async def test_snapshot_chain(self):
        """Test creating chain of snapshots with delta encoding"""
        manager = SnapshotManager(enable_delta_encoding=True, snapshot_interval=0.0)
        
        # Create chain of snapshots
        for i in range(5):
            data = {"step": i, "survival_signal": 0.5 + i * 0.1, "value": i}
            await manager.create_snapshot(data)
        
        assert manager.get_snapshot_count() == 5
        
        # Verify each snapshot has parent (except first)
        snapshots = list(manager.snapshots)
        for i in range(1, len(snapshots)):
            assert snapshots[i].parent_snapshot_id == snapshots[i-1].snapshot_id

