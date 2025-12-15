"""
Comprehensive test suite for Snapshot and Merkle modules

Tests all functionality to improve coverage from ~21-22% to 90%+.
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
from pulseos.persistence.merkle import (
    MerkleTree,
    MerkleNode,
    SnapshotIntegrity
)


class TestStateSnapshot:
    """Comprehensive tests for StateSnapshot"""
    
    def test_initialization_no_parent(self):
        """Test snapshot initialization without parent"""
        data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.5}}}
        snapshot = StateSnapshot(data, enable_delta_encoding=False, enable_compression=False)
        
        assert snapshot.snapshot_id is not None
        assert snapshot.step == 1
        assert snapshot.survival_signal == 0.8
        assert snapshot.agent_count == 1
        assert snapshot.parent_snapshot_id is None
    
    def test_initialization_with_parent(self):
        """Test snapshot initialization with parent"""
        parent_data = {"step": 0, "survival_signal": 0.7, "agents": {"agent1": {"state": 0.4}}}
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        current_data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.5}}}
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        assert snapshot.step == 1
        assert snapshot.parent_snapshot_id == parent.snapshot_id
    
    def test_delta_encoding_enabled(self):
        """Test delta encoding when enabled"""
        parent_data = {
            "step": 0,
            "survival_signal": 0.7,
            "agents": {"agent1": {"state": 0.4}, "agent2": {"state": 0.5}}
        }
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        # Only change one agent
        current_data = {
            "step": 1,
            "survival_signal": 0.7,  # Same
            "agents": {"agent1": {"state": 0.6}, "agent2": {"state": 0.5}}  # Only agent1 changed
        }
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        # Delta should be smaller than full snapshot
        assert snapshot.size_bytes > 0
        assert snapshot.parent_snapshot_id == parent.snapshot_id
    
    def test_delta_encoding_disabled(self):
        """Test delta encoding when disabled"""
        parent_data = {"step": 0, "survival_signal": 0.7}
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        current_data = {"step": 1, "survival_signal": 0.8}
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=False)
        
        assert snapshot.parent_snapshot_id is None
    
    def test_compression_enabled(self):
        """Test compression when enabled"""
        data = {"step": 1, "survival_signal": 0.8, "data": "x" * 1000}
        snapshot = StateSnapshot(data, enable_compression=True)
        
        assert snapshot.compressed_size_bytes <= snapshot.size_bytes
        assert snapshot.compressed_data != snapshot.raw_data
    
    def test_compression_disabled(self):
        """Test compression when disabled"""
        data = {"step": 1, "survival_signal": 0.8}
        snapshot = StateSnapshot(data, enable_compression=False)
        
        assert snapshot.compressed_size_bytes == snapshot.size_bytes
        assert snapshot.compressed_data == snapshot.raw_data
    
    def test_get_full_data_no_parent(self):
        """Test getting full data without parent"""
        data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.5}}}
        snapshot = StateSnapshot(data, enable_delta_encoding=False)
        
        full_data = snapshot.get_full_data()
        
        assert full_data["step"] == 1
        assert full_data["survival_signal"] == 0.8
        assert full_data["agents"]["agent1"]["state"] == 0.5
    
    def test_get_full_data_with_parent(self):
        """Test getting full data with parent snapshot"""
        parent_data = {"step": 0, "survival_signal": 0.7, "agents": {"agent1": {"state": 0.4}}}
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        current_data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.6}}}
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        full_data = snapshot.get_full_data(parent_snapshot=parent)
        
        assert full_data["step"] == 1
        assert full_data["survival_signal"] == 0.8
        assert full_data["agents"]["agent1"]["state"] == 0.6
    
    def test_get_full_data_compressed(self):
        """Test getting full data from compressed snapshot"""
        data = {"step": 1, "survival_signal": 0.8}
        snapshot = StateSnapshot(data, enable_compression=True)
        
        full_data = snapshot.get_full_data()
        
        assert full_data["step"] == 1
        assert full_data["survival_signal"] == 0.8
    
    def test_get_metadata(self):
        """Test getting snapshot metadata"""
        data = {"step": 42, "survival_signal": 0.85, "agents": {"a1": {}, "a2": {}}}
        snapshot = StateSnapshot(data)
        
        metadata = snapshot.get_metadata()
        
        assert isinstance(metadata, SnapshotMetadata)
        assert metadata.snapshot_id == snapshot.snapshot_id
        assert metadata.step == 42
        assert metadata.survival_signal == 0.85
        assert metadata.agent_count == 2
        assert metadata.size_bytes == snapshot.size_bytes
        assert metadata.compressed_size_bytes == snapshot.compressed_size_bytes
    
    def test_get_compression_ratio(self):
        """Test getting compression ratio"""
        data = {"step": 1, "survival_signal": 0.8}
        snapshot = StateSnapshot(data, enable_compression=True)
        
        ratio = snapshot.get_compression_ratio()
        
        assert 0.0 <= ratio <= 1.0
    
    def test_get_compression_ratio_zero_size(self):
        """Test compression ratio with zero size"""
        data = {}
        snapshot = StateSnapshot(data, enable_compression=False)
        
        # Force size_bytes to 0 for test
        snapshot.size_bytes = 0
        ratio = snapshot.get_compression_ratio()
        
        assert ratio == 1.0
    
    def test_delta_encoding_nested_dicts(self):
        """Test delta encoding with nested dictionaries"""
        parent_data = {
            "step": 0,
            "agents": {
                "agent1": {"state": 0.4, "metadata": {"version": 1}},
                "agent2": {"state": 0.5}
            }
        }
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        current_data = {
            "step": 1,
            "agents": {
                "agent1": {"state": 0.6, "metadata": {"version": 2}},  # Changed
                "agent2": {"state": 0.5}  # Unchanged
            }
        }
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        full_data = snapshot.get_full_data(parent_snapshot=parent)
        assert full_data["agents"]["agent1"]["state"] == 0.6
        assert full_data["agents"]["agent1"]["metadata"]["version"] == 2
        assert full_data["agents"]["agent2"]["state"] == 0.5
    
    def test_delta_encoding_new_keys(self):
        """Test delta encoding with new keys"""
        parent_data = {"step": 0, "agents": {"agent1": {"state": 0.4}}}
        parent = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        current_data = {"step": 1, "agents": {"agent1": {"state": 0.4}, "agent2": {"state": 0.5}}
        snapshot = StateSnapshot(current_data, parent_snapshot=parent, enable_delta_encoding=True)
        
        full_data = snapshot.get_full_data(parent_snapshot=parent)
        assert "agent2" in full_data["agents"]
        assert full_data["agents"]["agent2"]["state"] == 0.5
    
    def test_snapshot_id_uniqueness(self):
        """Test that snapshot IDs are unique"""
        data = {"step": 1}
        snapshot1 = StateSnapshot(data)
        time.sleep(0.001)  # Small delay to ensure different timestamp
        snapshot2 = StateSnapshot(data)
        
        assert snapshot1.snapshot_id != snapshot2.snapshot_id
    
    def test_empty_data(self):
        """Test snapshot with empty data"""
        snapshot = StateSnapshot({}, enable_delta_encoding=False)
        
        assert snapshot.step == 0
        assert snapshot.survival_signal == 0.0
        assert snapshot.agent_count == 0
        assert snapshot.size_bytes > 0  # Should still serialize
    
    def test_large_data(self):
        """Test snapshot with large data"""
        large_data = {
            "step": 1,
            "agents": {f"agent_{i}": {"state": i * 0.1} for i in range(1000)},
            "large_field": "x" * 10000
        }
        snapshot = StateSnapshot(large_data, enable_delta_encoding=False)
        
        assert snapshot.agent_count == 1000
        assert snapshot.size_bytes > 10000


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
    
    @pytest.mark.asyncio
    async def test_create_snapshot(self):
        """Test creating a snapshot"""
        manager = SnapshotManager(snapshot_interval=0.0)  # No interval restriction
        
        data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {}}}
        snapshot = await manager.create_snapshot(data)
        
        assert snapshot is not None
        assert snapshot.step == 1
        assert manager.get_snapshot_count() == 1
        assert manager.total_snapshots_created == 1
    
    @pytest.mark.asyncio
    async def test_create_snapshot_with_interval(self):
        """Test snapshot creation respects interval"""
        manager = SnapshotManager(snapshot_interval=0.1)
        
        data1 = {"step": 1, "survival_signal": 0.8}
        snapshot1 = await manager.create_snapshot(data1)
        
        # Try to create immediately (should return same snapshot)
        data2 = {"step": 2, "survival_signal": 0.9}
        snapshot2 = await manager.create_snapshot(data2)
        
        # Should return same snapshot if interval not met
        assert snapshot1.snapshot_id == snapshot2.snapshot_id
        
        # Wait for interval
        await asyncio.sleep(0.15)
        snapshot3 = await manager.create_snapshot(data2)
        
        # Should create new snapshot
        assert snapshot3.snapshot_id != snapshot1.snapshot_id
    
    @pytest.mark.asyncio
    async def test_create_snapshot_delta_encoding(self):
        """Test snapshot creation with delta encoding"""
        manager = SnapshotManager(
            snapshot_interval=0.0,
            enable_delta_encoding=True
        )
        
        parent_data = {"step": 0, "survival_signal": 0.7, "agents": {"agent1": {"state": 0.4}}}
        parent = await manager.create_snapshot(parent_data)
        
        current_data = {"step": 1, "survival_signal": 0.8, "agents": {"agent1": {"state": 0.5}}}
        snapshot = await manager.create_snapshot(current_data)
        
        assert snapshot.parent_snapshot_id == parent.snapshot_id
    
    @pytest.mark.asyncio
    async def test_max_snapshots_limit(self):
        """Test that max_snapshots limit is respected"""
        manager = SnapshotManager(
            snapshot_interval=0.0,
            max_snapshots=5
        )
        
        # Create more than max
        for i in range(10):
            data = {"step": i, "survival_signal": 0.8}
            await manager.create_snapshot(data)
        
        # Should only keep max_snapshots
        assert manager.get_snapshot_count() == 5
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot(self):
        """Test finding best recovery snapshot"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        # Create snapshots with varying survival signals
        for i, signal in enumerate([0.2, 0.5, 0.8, 0.3, 0.9]):
            data = {"step": i, "survival_signal": signal, "agents": {}}
            await manager.create_snapshot(data)
        
        # Find best recovery snapshot
        best = await manager.find_best_recovery_snapshot(min_survival_signal=0.3)
        
        assert best is not None
        assert best.survival_signal == 0.9  # Highest above threshold
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot_no_candidates(self):
        """Test finding recovery snapshot when no candidates meet threshold"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        # Create snapshots all below threshold
        for i, signal in enumerate([0.1, 0.2, 0.25]):
            data = {"step": i, "survival_signal": signal, "agents": {}}
            await manager.create_snapshot(data)
        
        # Should fallback to most recent
        best = await manager.find_best_recovery_snapshot(min_survival_signal=0.3)
        
        assert best is not None
        assert best.survival_signal == 0.25  # Most recent
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot_empty(self):
        """Test finding recovery snapshot when no snapshots exist"""
        manager = SnapshotManager()
        
        best = await manager.find_best_recovery_snapshot()
        
        assert best is None
    
    def test_get_snapshot(self):
        """Test getting snapshot by ID"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        data = {"step": 1, "survival_signal": 0.8, "agents": {}}
        snapshot = asyncio.run(manager.create_snapshot(data))
        
        retrieved = manager.get_snapshot(snapshot.snapshot_id)
        
        assert retrieved is not None
        assert retrieved.snapshot_id == snapshot.snapshot_id
    
    def test_get_snapshot_nonexistent(self):
        """Test getting non-existent snapshot"""
        manager = SnapshotManager()
        
        retrieved = manager.get_snapshot("nonexistent_id")
        
        assert retrieved is None
    
    def test_get_statistics_empty(self):
        """Test getting statistics with no snapshots"""
        manager = SnapshotManager()
        
        stats = manager.get_statistics()
        
        assert stats["snapshot_count"] == 0
        assert stats["total_created"] == 0
        assert stats["total_rollbacks"] == 0
        assert stats["average_size_bytes"] == 0
        assert stats["average_compression_ratio"] == 0.0
    
    def test_get_statistics_with_snapshots(self):
        """Test getting statistics with snapshots"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        for i in range(5):
            data = {"step": i, "survival_signal": 0.8, "agents": {}}
            asyncio.run(manager.create_snapshot(data))
        
        stats = manager.get_statistics()
        
        assert stats["snapshot_count"] == 5
        assert stats["total_created"] == 5
        assert stats["average_size_bytes"] > 0
        assert "total_storage_bytes" in stats
    
    def test_clear_snapshots(self):
        """Test clearing all snapshots"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        for i in range(5):
            data = {"step": i, "survival_signal": 0.8, "agents": {}}
            asyncio.run(manager.create_snapshot(data))
        
        assert manager.get_snapshot_count() == 5
        
        manager.clear_snapshots()
        
        assert manager.get_snapshot_count() == 0
        assert len(manager.snapshot_index) == 0


class TestMerkleTree:
    """Comprehensive tests for MerkleTree"""
    
    def test_initialization_single_block(self):
        """Test Merkle tree with single block"""
        blocks = [b"single block"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.root.data == blocks[0]
        assert tree.get_root_hash() is not None
    
    def test_initialization_multiple_blocks(self):
        """Test Merkle tree with multiple blocks"""
        blocks = [b"block1", b"block2", b"block3", b"block4"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.get_root_hash() is not None
    
    def test_initialization_odd_blocks(self):
        """Test Merkle tree with odd number of blocks"""
        blocks = [b"block1", b"block2", b"block3"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.verify(blocks) is True
    
    def test_verify_correct_blocks(self):
        """Test verification with correct blocks"""
        blocks = [b"block1", b"block2", b"block3"]
        tree = MerkleTree(blocks)
        
        assert tree.verify(blocks) is True
    
    def test_verify_modified_blocks(self):
        """Test verification with modified blocks"""
        blocks1 = [b"block1", b"block2"]
        blocks2 = [b"block1", b"modified"]
        
        tree = MerkleTree(blocks1)
        
        assert tree.verify(blocks1) is True
        assert tree.verify(blocks2) is False
    
    def test_verify_different_length(self):
        """Test verification with different length"""
        blocks1 = [b"block1", b"block2"]
        blocks2 = [b"block1"]
        
        tree = MerkleTree(blocks1)
        
        assert tree.verify(blocks2) is False
    
    def test_get_root_hash_hex(self):
        """Test getting root hash as hex string"""
        blocks = [b"test data"]
        tree = MerkleTree(blocks)
        
        hex_hash = tree.get_root_hash_hex()
        
        assert isinstance(hex_hash, str)
        assert len(hex_hash) == 64  # SHA-256 hex length
    
    def test_get_proof(self):
        """Test getting Merkle proof"""
        blocks = [b"block1", b"block2", b"block3", b"block4"]
        tree = MerkleTree(blocks)
        
        proof = tree.get_proof(0)
        
        assert isinstance(proof, list)
    
    def test_tree_structure(self):
        """Test that tree structure is correct"""
        blocks = [b"1", b"2", b"3", b"4"]
        tree = MerkleTree(blocks)
        
        # Root should have children
        assert tree.root is not None
        assert tree.root.hash_value is not None
    
    def test_deterministic_hashing(self):
        """Test that same blocks produce same hash"""
        blocks = [b"block1", b"block2"]
        
        tree1 = MerkleTree(blocks)
        tree2 = MerkleTree(blocks)
        
        assert tree1.get_root_hash() == tree2.get_root_hash()
    
    def test_empty_blocks(self):
        """Test with empty blocks"""
        blocks = [b"", b"", b""]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.verify(blocks) is True


class TestSnapshotIntegrity:
    """Comprehensive tests for SnapshotIntegrity"""
    
    def test_initialization(self):
        """Test snapshot integrity initialization"""
        integrity = SnapshotIntegrity()
        
        assert len(integrity.snapshot_hashes) == 0
    
    def test_compute_snapshot_hash(self):
        """Test computing snapshot hash"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        hash_value = integrity.compute_snapshot_hash(snapshot_data)
        
        assert isinstance(hash_value, bytes)
        assert len(hash_value) == 32  # SHA-256 digest length
    
    def test_compute_snapshot_hash_large_data(self):
        """Test computing hash for large snapshot"""
        integrity = SnapshotIntegrity()
        
        large_data = {
            "step": 1,
            "agents": {f"agent_{i}": {"state": i * 0.1} for i in range(1000)},
            "large_field": "x" * 10000
        }
        hash_value = integrity.compute_snapshot_hash(large_data)
        
        assert isinstance(hash_value, bytes)
        assert len(hash_value) == 32
    
    def test_register_snapshot(self):
        """Test registering snapshot"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        integrity.register_snapshot("snapshot1", snapshot_data)
        
        assert "snapshot1" in integrity.snapshot_hashes
        assert isinstance(integrity.snapshot_hashes["snapshot1"], bytes)
    
    def test_verify_snapshot_valid(self):
        """Test verifying valid snapshot"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        integrity.register_snapshot("snapshot1", snapshot_data)
        
        assert integrity.verify_snapshot("snapshot1", snapshot_data) is True
    
    def test_verify_snapshot_modified(self):
        """Test verifying modified snapshot"""
        integrity = SnapshotIntegrity()
        
        original_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        integrity.register_snapshot("snapshot1", original_data)
        
        modified_data = {"step": 1, "agents": {"agent1": {"state": 0.6}}}
        
        assert integrity.verify_snapshot("snapshot1", modified_data) is False
    
    def test_verify_snapshot_unregistered(self):
        """Test verifying unregistered snapshot"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        
        assert integrity.verify_snapshot("nonexistent", snapshot_data) is False
    
    def test_multiple_snapshots(self):
        """Test handling multiple snapshots"""
        integrity = SnapshotIntegrity()
        
        for i in range(5):
            snapshot_data = {"step": i, "agents": {"agent1": {"state": 0.5}}}
            integrity.register_snapshot(f"snapshot_{i}", snapshot_data)
        
        assert len(integrity.snapshot_hashes) == 5
        
        # Verify each
        for i in range(5):
            snapshot_data = {"step": i, "agents": {"agent1": {"state": 0.5}}}
            assert integrity.verify_snapshot(f"snapshot_{i}", snapshot_data) is True
    
    def test_hash_consistency(self):
        """Test that hash is consistent for same data"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        
        hash1 = integrity.compute_snapshot_hash(snapshot_data)
        hash2 = integrity.compute_snapshot_hash(snapshot_data)
        
        assert hash1 == hash2
    
    def test_hash_different_data(self):
        """Test that different data produces different hash"""
        integrity = SnapshotIntegrity()
        
        data1 = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        data2 = {"step": 2, "agents": {"agent1": {"state": 0.5}}}
        
        hash1 = integrity.compute_snapshot_hash(data1)
        hash2 = integrity.compute_snapshot_hash(data2)
        
        assert hash1 != hash2

