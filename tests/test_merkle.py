"""
Comprehensive test suite for merkle.py

Tests all functionality to achieve 90%+ coverage.
"""

import pytest
import pickle
from pulseos.persistence.merkle import (
    MerkleTree,
    MerkleNode,
    SnapshotIntegrity
)


class TestMerkleNode:
    """Tests for MerkleNode dataclass"""
    
    def test_creation_leaf(self):
        """Test creating a leaf node"""
        node = MerkleNode(
            hash_value=b"hash123",
            data=b"data123"
        )
        
        assert node.hash_value == b"hash123"
        assert node.data == b"data123"
        assert node.left is None
        assert node.right is None
    
    def test_creation_internal(self):
        """Test creating an internal node"""
        left = MerkleNode(hash_value=b"left", data=b"left_data")
        right = MerkleNode(hash_value=b"right", data=b"right_data")
        
        node = MerkleNode(
            hash_value=b"parent",
            left=left,
            right=right
        )
        
        assert node.hash_value == b"parent"
        assert node.left == left
        assert node.right == right
        assert node.data is None


class TestMerkleTree:
    """Comprehensive tests for MerkleTree"""
    
    def test_initialization_single_block(self):
        """Test initialization with single block"""
        blocks = [b"single block"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.root.data == b"single block"
        assert tree.data_blocks == blocks
    
    def test_initialization_multiple_blocks(self):
        """Test initialization with multiple blocks"""
        blocks = [b"block1", b"block2", b"block3", b"block4"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.data_blocks == blocks
    
    def test_initialization_odd_blocks(self):
        """Test initialization with odd number of blocks"""
        blocks = [b"block1", b"block2", b"block3"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.data_blocks == blocks
    
    def test_get_root_hash(self):
        """Test getting root hash"""
        blocks = [b"block1", b"block2"]
        tree = MerkleTree(blocks)
        
        root_hash = tree.get_root_hash()
        
        assert isinstance(root_hash, bytes)
        assert len(root_hash) == 32  # SHA-256 produces 32 bytes
    
    def test_get_root_hash_hex(self):
        """Test getting root hash as hex string"""
        blocks = [b"block1", b"block2"]
        tree = MerkleTree(blocks)
        
        root_hash_hex = tree.get_root_hash_hex()
        
        assert isinstance(root_hash_hex, str)
        assert len(root_hash_hex) == 64  # 32 bytes * 2 hex chars
    
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
    
    def test_verify_different_count(self):
        """Test verification with different block count"""
        blocks1 = [b"block1", b"block2"]
        blocks2 = [b"block1"]
        
        tree = MerkleTree(blocks1)
        
        assert tree.verify(blocks2) is False
    
    def test_verify_empty_blocks(self):
        """Test verification with empty blocks"""
        blocks = [b"block1", b"block2"]
        empty_blocks = []
        
        tree = MerkleTree(blocks)
        
        assert tree.verify(empty_blocks) is False
    
    def test_get_proof_single_block(self):
        """Test getting proof for single block"""
        blocks = [b"single block"]
        tree = MerkleTree(blocks)
        
        proof = tree.get_proof(0)
        
        # With single block, proof should be empty (no siblings)
        assert isinstance(proof, list)
    
    def test_get_proof_two_blocks(self):
        """Test getting proof for two blocks"""
        blocks = [b"block1", b"block2"]
        tree = MerkleTree(blocks)
        
        proof = tree.get_proof(0)
        
        assert isinstance(proof, list)
        # Proof should contain sibling hash
    
    def test_get_proof_multiple_blocks(self):
        """Test getting proof for multiple blocks"""
        blocks = [b"block1", b"block2", b"block3", b"block4"]
        tree = MerkleTree(blocks)
        
        proof = tree.get_proof(0)
        
        assert isinstance(proof, list)
    
    def test_tree_structure_two_blocks(self):
        """Test tree structure with two blocks"""
        blocks = [b"block1", b"block2"]
        tree = MerkleTree(blocks)
        
        # Root should have left and right children
        assert tree.root is not None
        assert tree.root.left is not None
        assert tree.root.right is not None
    
    def test_tree_structure_four_blocks(self):
        """Test tree structure with four blocks"""
        blocks = [b"block1", b"block2", b"block3", b"block4"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        # Should have two children at root level
        assert tree.root.left is not None
        assert tree.root.right is not None
    
    def test_tree_structure_three_blocks(self):
        """Test tree structure with three blocks (odd number)"""
        blocks = [b"block1", b"block2", b"block3"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        # Odd blocks should be handled correctly
    
    def test_hash_consistency(self):
        """Test hash consistency"""
        blocks = [b"block1", b"block2"]
        tree1 = MerkleTree(blocks)
        tree2 = MerkleTree(blocks)
        
        # Same blocks should produce same root hash
        assert tree1.get_root_hash() == tree2.get_root_hash()
    
    def test_hash_different_blocks(self):
        """Test hash differs for different blocks"""
        blocks1 = [b"block1", b"block2"]
        blocks2 = [b"block1", b"block3"]
        
        tree1 = MerkleTree(blocks1)
        tree2 = MerkleTree(blocks2)
        
        # Different blocks should produce different root hash
        assert tree1.get_root_hash() != tree2.get_root_hash()
    
    def test_hash_order_matters(self):
        """Test that block order matters for hash"""
        blocks1 = [b"block1", b"block2"]
        blocks2 = [b"block2", b"block1"]
        
        tree1 = MerkleTree(blocks1)
        tree2 = MerkleTree(blocks2)
        
        # Different order should produce different hash
        assert tree1.get_root_hash() != tree2.get_root_hash()
    
    def test_large_number_of_blocks(self):
        """Test tree with large number of blocks"""
        blocks = [f"block{i}".encode() for i in range(100)]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.verify(blocks) is True
    
    def test_empty_block(self):
        """Test tree with empty block"""
        blocks = [b"", b"block2"]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.verify(blocks) is True
    
    def test_unicode_blocks(self):
        """Test tree with unicode blocks"""
        blocks = ["block1_中文".encode('utf-8'), "block2_日本語".encode('utf-8')]
        tree = MerkleTree(blocks)
        
        assert tree.root is not None
        assert tree.verify(blocks) is True


class TestSnapshotIntegrity:
    """Comprehensive tests for SnapshotIntegrity"""
    
    def test_initialization(self):
        """Test initialization"""
        integrity = SnapshotIntegrity()
        
        assert len(integrity.snapshot_hashes) == 0
    
    def test_compute_snapshot_hash(self):
        """Test computing snapshot hash"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        hash_value = integrity.compute_snapshot_hash(snapshot_data)
        
        assert isinstance(hash_value, bytes)
        assert len(hash_value) == 32  # SHA-256
    
    def test_compute_snapshot_hash_consistency(self):
        """Test hash consistency for same data"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        hash1 = integrity.compute_snapshot_hash(snapshot_data)
        hash2 = integrity.compute_snapshot_hash(snapshot_data)
        
        assert hash1 == hash2
    
    def test_compute_snapshot_hash_different_data(self):
        """Test hash differs for different data"""
        integrity = SnapshotIntegrity()
        
        data1 = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        data2 = {"step": 1, "agents": {"agent1": {"state": 0.6}}}
        
        hash1 = integrity.compute_snapshot_hash(data1)
        hash2 = integrity.compute_snapshot_hash(data2)
        
        assert hash1 != hash2
    
    def test_compute_snapshot_hash_large_data(self):
        """Test hash computation with large data"""
        integrity = SnapshotIntegrity()
        
        large_data = {
            "step": 1,
            "agents": {f"agent_{i}": {"state": i * 0.1} for i in range(1000)}
        }
        
        hash_value = integrity.compute_snapshot_hash(large_data)
        
        assert isinstance(hash_value, bytes)
        assert len(hash_value) == 32
    
    def test_register_snapshot(self):
        """Test registering a snapshot"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        integrity.register_snapshot("snapshot1", snapshot_data)
        
        assert "snapshot1" in integrity.snapshot_hashes
        assert isinstance(integrity.snapshot_hashes["snapshot1"], bytes)
    
    def test_register_multiple_snapshots(self):
        """Test registering multiple snapshots"""
        integrity = SnapshotIntegrity()
        
        for i in range(5):
            snapshot_data = {"step": i, "agents": {"agent1": {"state": i * 0.1}}}
            integrity.register_snapshot(f"snapshot{i}", snapshot_data)
        
        assert len(integrity.snapshot_hashes) == 5
    
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
        modified_data = {"step": 1, "agents": {"agent1": {"state": 0.6}}}
        
        integrity.register_snapshot("snapshot1", original_data)
        
        assert integrity.verify_snapshot("snapshot1", original_data) is True
        assert integrity.verify_snapshot("snapshot1", modified_data) is False
    
    def test_verify_snapshot_nonexistent(self):
        """Test verifying nonexistent snapshot"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        
        assert integrity.verify_snapshot("nonexistent", snapshot_data) is False
    
    def test_verify_snapshot_empty_data(self):
        """Test verifying snapshot with empty data"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {}
        integrity.register_snapshot("snapshot1", snapshot_data)
        
        assert integrity.verify_snapshot("snapshot1", snapshot_data) is True
    
    def test_verify_snapshot_nested_data(self):
        """Test verifying snapshot with nested data"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {
            "step": 1,
            "agents": {
                "agent1": {
                    "state": 0.5,
                    "nested": {
                        "value": 10,
                        "deep": {"key": "value"}
                    }
                }
            }
        }
        
        integrity.register_snapshot("snapshot1", snapshot_data)
        
        assert integrity.verify_snapshot("snapshot1", snapshot_data) is True
        
        # Modify nested value
        modified_data = snapshot_data.copy()
        modified_data["agents"]["agent1"]["nested"]["value"] = 20
        
        assert integrity.verify_snapshot("snapshot1", modified_data) is False
    
    def test_hash_chunking(self):
        """Test that large snapshots are chunked correctly"""
        integrity = SnapshotIntegrity()
        
        # Create data larger than 1KB to test chunking
        large_data = {
            "step": 1,
            "data": "x" * 2000  # 2KB string
        }
        
        hash_value = integrity.compute_snapshot_hash(large_data)
        
        assert isinstance(hash_value, bytes)
        assert len(hash_value) == 32
    
    def test_register_overwrite(self):
        """Test registering snapshot with same ID overwrites"""
        integrity = SnapshotIntegrity()
        
        data1 = {"step": 1, "value": 10}
        data2 = {"step": 1, "value": 20}
        
        integrity.register_snapshot("snapshot1", data1)
        hash1 = integrity.snapshot_hashes["snapshot1"]
        
        integrity.register_snapshot("snapshot1", data2)
        hash2 = integrity.snapshot_hashes["snapshot1"]
        
        assert hash1 != hash2
        assert integrity.verify_snapshot("snapshot1", data2) is True
        assert integrity.verify_snapshot("snapshot1", data1) is False

