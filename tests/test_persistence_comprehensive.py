"""
Comprehensive test suite for Snapshot and Merkle modules

Tests all functionality to achieve 90%+ coverage for snapshot.py and merkle.py.
"""

import pytest
import asyncio
import time
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
    
    def test_initialization_full_snapshot(self):
        """Test creating full snapshot without delta encoding"""
        data = {
            "step": 10,
            "survival_signal": 0.85,
            "agents": {
                "agent1": {"state": 0.5, "performance": 0.9},
                "agent2": {"state": 0.6, "performance": 0.8}
            }
        }
        
        snapshot = StateSnapshot(
            data,
            enable_delta_encoding=False,
            enable_compression=False
        )
        
        assert snapshot.step == 10
        assert snapshot.survival_signal == 0.85
        assert snapshot.agent_count == 2
        assert snapshot.parent_snapshot_id is None
        assert snapshot.size_bytes > 0
        assert snapshot.compressed_size_bytes == snapshot.size_bytes
    
    def test_initialization_with_compression(self):
        """Test snapshot with compression enabled"""
        data = {
            "step": 1,
            "agents": {"agent1": {"state": 0.5}}
        }
        
        snapshot = StateSnapshot(
            data,
            enable_delta_encoding=False,
            enable_compression=True
        )
        
        assert snapshot.compressed_size_bytes <= snapshot.size_bytes
        assert snapshot.compressed_size_bytes > 0
    
    def test_delta_encoding_no_parent(self):
        """Test delta encoding without parent (should create full snapshot)"""
        data = {
            "step": 1,
            "agents": {"agent1": {"state": 0.5}}
        }
        
        snapshot = StateSnapshot(
            data,
            parent_snapshot=None,
            enable_delta_encoding=True
        )
        
        assert snapshot.parent_snapshot_id is None
        full_data = snapshot.get_full_data()
        assert full_data["step"] == 1
    
    def test_delta_encoding_with_parent(self):
        """Test delta encoding with parent snapshot"""
        parent_data = {
            "step": 1,
            "agents": {
                "agent1": {"state": 0.5, "performance": 0.8},
                "agent2": {"state": 0.6, "performance": 0.7}
            }
        }
        
        parent_snapshot = StateSnapshot(
            parent_data,
            enable_delta_encoding=False
        )
        
        # Create delta snapshot with changes
        current_data = {
            "step": 2,
            "agents": {
                "agent1": {"state": 0.6, "performance": 0.9},  # Changed
                "agent2": {"state": 0.6, "performance": 0.7}   # Unchanged
            }
        }
        
        delta_snapshot = StateSnapshot(
            current_data,
            parent_snapshot=parent_snapshot,
            enable_delta_encoding=True
        )
        
        assert delta_snapshot.parent_snapshot_id == parent_snapshot.snapshot_id
        assert delta_snapshot.size_bytes < parent_snapshot.size_bytes
        
        # Verify full data reconstruction
        full_data = delta_snapshot.get_full_data(parent_snapshot)
        assert full_data["step"] == 2
        assert full_data["agents"]["agent1"]["state"] == 0.6
        assert full_data["agents"]["agent2"]["state"] == 0.6
    
    def test_delta_encoding_nested_dicts(self):
        """Test delta encoding with nested dictionaries"""
        parent_data = {
            "step": 1,
            "config": {
                "learning_rate": 0.01,
                "exploration": {
                    "epsilon": 0.1,
                    "decay": 0.99
                }
            }
        }
        
        parent_snapshot = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        current_data = {
            "step": 2,
            "config": {
                "learning_rate": 0.02,  # Changed
                "exploration": {
                    "epsilon": 0.1,  # Unchanged
                    "decay": 0.98    # Changed
                }
            }
        }
        
        delta_snapshot = StateSnapshot(
            current_data,
            parent_snapshot=parent_snapshot,
            enable_delta_encoding=True
        )
        
        full_data = delta_snapshot.get_full_data(parent_snapshot)
        assert full_data["config"]["learning_rate"] == 0.02
        assert full_data["config"]["exploration"]["epsilon"] == 0.1
        assert full_data["config"]["exploration"]["decay"] == 0.98
    
    def test_delta_encoding_new_keys(self):
        """Test delta encoding with new keys added"""
        parent_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        parent_snapshot = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        current_data = {
            "step": 2,
            "agents": {
                "agent1": {"state": 0.5},
                "agent2": {"state": 0.6}  # New agent
            }
        }
        
        delta_snapshot = StateSnapshot(
            current_data,
            parent_snapshot=parent_snapshot,
            enable_delta_encoding=True
        )
        
        full_data = delta_snapshot.get_full_data(parent_snapshot)
        assert "agent2" in full_data["agents"]
    
    def test_delta_encoding_removed_keys(self):
        """Test delta encoding with removed keys"""
        parent_data = {
            "step": 1,
            "agents": {
                "agent1": {"state": 0.5},
                "agent2": {"state": 0.6}
            }
        }
        parent_snapshot = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        current_data = {
            "step": 2,
            "agents": {
                "agent1": {"state": 0.5}  # agent2 removed
            }
        }
        
        delta_snapshot = StateSnapshot(
            current_data,
            parent_snapshot=parent_snapshot,
            enable_delta_encoding=True
        )
        
        full_data = delta_snapshot.get_full_data(parent_snapshot)
        assert "agent2" not in full_data["agents"]
    
    def test_get_full_data_no_compression(self):
        """Test getting full data without compression"""
        data = {"step": 1, "value": "test"}
        snapshot = StateSnapshot(
            data,
            enable_delta_encoding=False,
            enable_compression=False
        )
        
        full_data = snapshot.get_full_data()
        assert full_data["step"] == 1
        assert full_data["value"] == "test"
    
    def test_get_full_data_with_compression(self):
        """Test getting full data with compression"""
        data = {"step": 1, "value": "test"}
        snapshot = StateSnapshot(
            data,
            enable_delta_encoding=False,
            enable_compression=True
        )
        
        full_data = snapshot.get_full_data()
        assert full_data["step"] == 1
        assert full_data["value"] == "test"
    
    def test_get_metadata(self):
        """Test getting snapshot metadata"""
        data = {
            "step": 42,
            "survival_signal": 0.75,
            "agents": {"agent1": {}, "agent2": {}}
        }
        
        snapshot = StateSnapshot(data, enable_delta_encoding=False)
        metadata = snapshot.get_metadata()
        
        assert isinstance(metadata, SnapshotMetadata)
        assert metadata.step == 42
        assert metadata.survival_signal == 0.75
        assert metadata.agent_count == 2
        assert metadata.size_bytes > 0
        assert metadata.snapshot_id == snapshot.snapshot_id
    
    def test_get_compression_ratio(self):
        """Test getting compression ratio"""
        data = {"step": 1, "data": "x" * 1000}
        
        snapshot_no_compression = StateSnapshot(
            data,
            enable_compression=False
        )
        assert snapshot_no_compression.get_compression_ratio() == 1.0
        
        snapshot_with_compression = StateSnapshot(
            data,
            enable_compression=True
        )
        ratio = snapshot_with_compression.get_compression_ratio()
        assert 0 < ratio <= 1.0
    
    def test_get_compression_ratio_zero_size(self):
        """Test compression ratio with zero size"""
        snapshot = StateSnapshot({}, enable_delta_encoding=False)
        # Should handle edge case gracefully
        assert snapshot.get_compression_ratio() >= 0
    
    def test_snapshot_id_uniqueness(self):
        """Test that snapshot IDs are unique"""
        data = {"step": 1}
        
        snapshot1 = StateSnapshot(data)
        time.sleep(0.001)  # Small delay
        snapshot2 = StateSnapshot(data)
        
        assert snapshot1.snapshot_id != snapshot2.snapshot_id


class TestSnapshotManager:
    """Comprehensive tests for SnapshotManager"""
    
    @pytest.mark.asyncio
    async def test_create_snapshot_first(self):
        """Test creating first snapshot"""
        manager = SnapshotManager(
            snapshot_interval=0.0,
            max_snapshots=10
        )
        
        data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        snapshot = await manager.create_snapshot(data)
        
        assert snapshot is not None
        assert manager.get_snapshot_count() == 1
        assert manager.total_snapshots_created == 1
    
    @pytest.mark.asyncio
    async def test_create_snapshot_interval(self):
        """Test snapshot creation respects interval"""
        manager = SnapshotManager(
            snapshot_interval=0.1,
            max_snapshots=10
        )
        
        data = {"step": 1}
        
        # Create first snapshot
        snapshot1 = await manager.create_snapshot(data)
        
        # Try to create immediately (should return same snapshot)
        snapshot2 = await manager.create_snapshot(data)
        
        assert snapshot1.snapshot_id == snapshot2.snapshot_id
        
        # Wait for interval
        await asyncio.sleep(0.11)
        snapshot3 = await manager.create_snapshot(data)
        
        assert snapshot3.snapshot_id != snapshot1.snapshot_id
    
    @pytest.mark.asyncio
    async def test_create_snapshot_max_limit(self):
        """Test snapshot manager respects max_snapshots limit"""
        manager = SnapshotManager(
            snapshot_interval=0.0,
            max_snapshots=3
        )
        
        for i in range(5):
            data = {"step": i}
            await manager.create_snapshot(data)
        
        assert manager.get_snapshot_count() == 3
    
    @pytest.mark.asyncio
    async def test_create_snapshot_delta_encoding(self):
        """Test snapshot creation with delta encoding"""
        manager = SnapshotManager(
            snapshot_interval=0.0,
            enable_delta_encoding=True
        )
        
        # Create parent snapshot
        parent_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        parent_snapshot = await manager.create_snapshot(parent_data)
        
        # Create delta snapshot
        current_data = {"step": 2, "agents": {"agent1": {"state": 0.6}}}
        delta_snapshot = await manager.create_snapshot(current_data)
        
        assert delta_snapshot.parent_snapshot_id == parent_snapshot.snapshot_id
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot_empty(self):
        """Test finding recovery snapshot with no snapshots"""
        manager = SnapshotManager()
        
        snapshot = await manager.find_best_recovery_snapshot()
        assert snapshot is None
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot_below_threshold(self):
        """Test finding recovery snapshot below threshold"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        # Create snapshots with low survival signals
        for i in range(3):
            data = {"step": i, "survival_signal": 0.1 + i * 0.05}
            await manager.create_snapshot(data)
        
        # Should return most recent (fallback)
        snapshot = await manager.find_best_recovery_snapshot(min_survival_signal=0.3)
        assert snapshot is not None
        assert snapshot == manager.snapshots[-1]
    
    @pytest.mark.asyncio
    async def test_find_best_recovery_snapshot_above_threshold(self):
        """Test finding recovery snapshot above threshold"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        # Create snapshots with varying survival signals
        signals = [0.2, 0.5, 0.3, 0.9, 0.4]
        for i, signal in enumerate(signals):
            data = {"step": i, "survival_signal": signal}
            await manager.create_snapshot(data)
        
        snapshot = await manager.find_best_recovery_snapshot(min_survival_signal=0.3)
        assert snapshot is not None
        assert snapshot.survival_signal == 0.9  # Highest above threshold
    
    @pytest.mark.asyncio
    async def test_get_snapshot_by_id(self):
        """Test getting snapshot by ID"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        data = {"step": 1}
        created_snapshot = await manager.create_snapshot(data)
        
        retrieved_snapshot = manager.get_snapshot(created_snapshot.snapshot_id)
        
        assert retrieved_snapshot is not None
        assert retrieved_snapshot.snapshot_id == created_snapshot.snapshot_id
    
    @pytest.mark.asyncio
    async def test_get_snapshot_nonexistent(self):
        """Test getting nonexistent snapshot"""
        manager = SnapshotManager()
        
        snapshot = manager.get_snapshot("nonexistent_id")
        assert snapshot is None
    
    def test_get_statistics_empty(self):
        """Test getting statistics with no snapshots"""
        manager = SnapshotManager()
        
        stats = manager.get_statistics()
        
        assert stats["snapshot_count"] == 0
        assert stats["total_created"] == 0
        assert stats["total_rollbacks"] == 0
        assert stats["average_size_bytes"] == 0
        assert stats["average_compression_ratio"] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_snapshots(self):
        """Test getting statistics with snapshots"""
        manager = SnapshotManager(snapshot_interval=0.0)
        
        for i in range(3):
            data = {"step": i}
            await manager.create_snapshot(data)
        
        stats = manager.get_statistics()
        
        assert stats["snapshot_count"] == 3
        assert stats["total_created"] == 3
        assert stats["average_size_bytes"] > 0
        assert stats["total_storage_bytes"] > 0
    
    def test_clear_snapshots(self):
        """Test clearing all snapshots"""
        manager = SnapshotManager()
        
        # Add some snapshots
        asyncio.run(manager.create_snapshot({"step": 1}))
        assert manager.get_snapshot_count() > 0
        
        manager.clear_snapshots()
        
        assert manager.get_snapshot_count() == 0
        assert len(manager.snapshot_index) == 0


class TestMerkleTree:
    """Comprehensive tests for MerkleTree"""
    
    def test_initialization_single_block(self):
        """Test Merkle tree with single block"""
        blocks = [b"test data"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.root.hash_value is not None
        assert tree.root.data == b"test data"
    
    def test_initialization_two_blocks(self):
        """Test Merkle tree with two blocks"""
        blocks = [b"block1", b"block2"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.root.left is not None
        assert tree.root.right is not None
    
    def test_initialization_three_blocks(self):
        """Test Merkle tree with odd number of blocks"""
        blocks = [b"block1", b"block2", b"block3"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.verify(blocks) is True
    
    def test_initialization_many_blocks(self):
        """Test Merkle tree with many blocks"""
        blocks = [f"block{i}".encode() for i in range(10)]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.verify(blocks) is True
    
    def test_get_root_hash(self):
        """Test getting root hash"""
        blocks = [b"test"]
        tree = MerkleTree(blocks)
        
        root_hash = tree.get_root_hash()
        
        assert isinstance(root_hash, bytes)
        assert len(root_hash) == 32  # SHA-256 produces 32 bytes
    
    def test_get_root_hash_hex(self):
        """Test getting root hash as hex string"""
        blocks = [b"test"]
        tree = MerkleTree(blocks)
        
        root_hash_hex = tree.get_root_hash_hex()
        
        assert isinstance(root_hash_hex, str)
        assert len(root_hash_hex) == 64  # 32 bytes * 2 hex chars
    
    def test_verify_same_blocks(self):
        """Test verification with same blocks"""
        blocks = [b"block1", b"block2", b"block3"]
        tree = MerkleTree(blocks)
        
        assert tree.verify(blocks) is True
    
    def test_verify_different_blocks(self):
        """Test verification with different blocks"""
        blocks1 = [b"block1", b"block2"]
        blocks2 = [b"block1", b"modified"]
        
        tree = MerkleTree(blocks1)
        
        assert tree.verify(blocks1) is True
        assert tree.verify(blocks2) is False
    
    def test_verify_different_count(self):
        """Test verification with different block count"""
        blocks1 = [b"block1", b"block2"]
        blocks2 = [b"block1"]
        
        tree = MerkleTree(blocks1)
        
        assert tree.verify(blocks2) is False
    
    def test_get_proof_single_block(self):
        """Test getting Merkle proof for single block"""
        blocks = [b"single"]
        tree = MerkleTree(blocks)
        
        proof = tree.get_proof(0)
        # With single block, proof should be empty or minimal
        assert isinstance(proof, list)
    
    def test_get_proof_multiple_blocks(self):
        """Test getting Merkle proof for multiple blocks"""
        blocks = [b"block1", b"block2", b"block3", b"block4"]
        tree = MerkleTree(blocks)
        
        proof = tree.get_proof(0)
        assert isinstance(proof, list)
        # Proof should contain sibling hashes
    
    def test_hash_consistency(self):
        """Test that hash computation is consistent"""
        blocks = [b"test data"]
        
        tree1 = MerkleTree(blocks)
        tree2 = MerkleTree(blocks)
        
        assert tree1.get_root_hash() == tree2.get_root_hash()
    
    def test_tree_structure(self):
        """Test Merkle tree structure"""
        blocks = [b"1", b"2", b"3", b"4"]
        tree = MerkleTree(blocks)
        
        # Verify tree structure
        assert tree.root is not None
        if tree.root.left is not None:
            assert tree.root.left.hash_value is not None
        if tree.root.right is not None:
            assert tree.root.right.hash_value is not None


class TestSnapshotIntegrity:
    """Comprehensive tests for SnapshotIntegrity"""
    
    def test_initialization(self):
        """Test SnapshotIntegrity initialization"""
        integrity = SnapshotIntegrity()
        
        assert len(integrity.snapshot_hashes) == 0
    
    def test_compute_snapshot_hash_small(self):
        """Test computing hash for small snapshot"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        hash_value = integrity.compute_snapshot_hash(snapshot_data)
        
        assert isinstance(hash_value, bytes)
        assert len(hash_value) == 32  # SHA-256
    
    def test_compute_snapshot_hash_large(self):
        """Test computing hash for large snapshot"""
        integrity = SnapshotIntegrity()
        
        # Create large snapshot
        snapshot_data = {
            "step": 1,
            "agents": {f"agent{i}": {"state": i * 0.1} for i in range(1000)}
        }
        
        hash_value = integrity.compute_snapshot_hash(snapshot_data)
        
        assert isinstance(hash_value, bytes)
        assert len(hash_value) == 32
    
    def test_compute_snapshot_hash_consistency(self):
        """Test that hash computation is consistent"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "value": "test"}
        
        hash1 = integrity.compute_snapshot_hash(snapshot_data)
        hash2 = integrity.compute_snapshot_hash(snapshot_data)
        
        assert hash1 == hash2
    
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
        
        snapshot_data = {"step": 1}
        assert integrity.verify_snapshot("nonexistent", snapshot_data) is False
    
    def test_multiple_snapshots(self):
        """Test managing multiple snapshots"""
        integrity = SnapshotIntegrity()
        
        for i in range(5):
            snapshot_data = {"step": i, "value": f"test{i}"}
            integrity.register_snapshot(f"snapshot{i}", snapshot_data)
        
        assert len(integrity.snapshot_hashes) == 5
        
        # Verify each snapshot
        for i in range(5):
            snapshot_data = {"step": i, "value": f"test{i}"}
            assert integrity.verify_snapshot(f"snapshot{i}", snapshot_data) is True

