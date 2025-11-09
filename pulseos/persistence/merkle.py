"""
Merkle Tree for Snapshot Integrity

Implements Merkle tree for snapshot integrity verification as specified in patent.
"""

import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MerkleNode:
    """Merkle tree node"""
    hash_value: bytes
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    data: Optional[bytes] = None


class MerkleTree:
    """
    Merkle tree for snapshot integrity verification.
    
    Implements patent-specified integrity checking for state snapshots.
    """
    
    def __init__(self, data_blocks: List[bytes]):
        """
        Build Merkle tree from data blocks.
        
        Args:
            data_blocks: List of data blocks to hash
        """
        self.data_blocks = data_blocks
        self.root = self._build_tree(data_blocks)
    
    def _hash(self, data: bytes) -> bytes:
        """Compute SHA-256 hash."""
        return hashlib.sha256(data).digest()
    
    def _build_tree(self, blocks: List[bytes]) -> MerkleNode:
        """Build Merkle tree recursively."""
        if len(blocks) == 1:
            return MerkleNode(
                hash_value=self._hash(blocks[0]),
                data=blocks[0]
            )
        
        # Hash each block
        nodes = [
            MerkleNode(hash_value=self._hash(block), data=block)
            for block in blocks
        ]
        
        # Build tree bottom-up
        while len(nodes) > 1:
            next_level = []
            
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    # Pair nodes
                    left = nodes[i]
                    right = nodes[i + 1]
                    combined_hash = self._hash(left.hash_value + right.hash_value)
                    parent = MerkleNode(
                        hash_value=combined_hash,
                        left=left,
                        right=right
                    )
                    next_level.append(parent)
                else:
                    # Odd node, promote
                    next_level.append(nodes[i])
            
            nodes = next_level
        
        return nodes[0]
    
    def get_root_hash(self) -> bytes:
        """Get root hash of Merkle tree."""
        return self.root.hash_value
    
    def get_root_hash_hex(self) -> str:
        """Get root hash as hexadecimal string."""
        return self.root.hash_value.hex()
    
    def verify(self, data_blocks: List[bytes]) -> bool:
        """
        Verify integrity of data blocks.
        
        Args:
            data_blocks: Data blocks to verify
            
        Returns:
            True if integrity is valid
        """
        if len(data_blocks) != len(self.data_blocks):
            return False
        
        # Rebuild tree and compare root hash
        verification_tree = MerkleTree(data_blocks)
        return verification_tree.get_root_hash() == self.get_root_hash()
    
    def get_proof(self, index: int) -> List[bytes]:
        """
        Get Merkle proof for a specific data block.
        
        Args:
            index: Index of data block
            
        Returns:
            List of hashes for Merkle proof
        """
        proof = []
        current = self.root
        
        # Traverse tree to find block
        blocks = self.data_blocks
        level_size = len(blocks)
        
        while level_size > 1:
            sibling_index = index ^ 1  # XOR to get sibling
            
            if sibling_index < level_size:
                # Get sibling hash
                if index < sibling_index:
                    proof.append(self._hash(blocks[sibling_index]))
                else:
                    proof.append(self._hash(blocks[sibling_index]))
            
            index //= 2
            level_size = (level_size + 1) // 2
        
        return proof


class SnapshotIntegrity:
    """
    Snapshot integrity manager using Merkle trees.
    
    Provides integrity verification for state snapshots as specified in patent.
    """
    
    def __init__(self):
        """Initialize snapshot integrity manager."""
        self.snapshot_hashes: Dict[str, bytes] = {}
    
    def compute_snapshot_hash(self, snapshot_data: Dict[str, Any]) -> bytes:
        """
        Compute Merkle root hash for snapshot.
        
        Args:
            snapshot_data: Snapshot data dictionary
            
        Returns:
            Merkle root hash
        """
        # Serialize snapshot data
        import pickle
        serialized = pickle.dumps(snapshot_data)
        
        # Create data blocks (chunk for large snapshots)
        block_size = 1024  # 1KB blocks
        blocks = [
            serialized[i:i + block_size]
            for i in range(0, len(serialized), block_size)
        ]
        
        # Build Merkle tree
        tree = MerkleTree(blocks)
        return tree.get_root_hash()
    
    def verify_snapshot(
        self,
        snapshot_id: str,
        snapshot_data: Dict[str, Any]
    ) -> bool:
        """
        Verify snapshot integrity.
        
        Args:
            snapshot_id: Snapshot identifier
            snapshot_data: Snapshot data to verify
            
        Returns:
            True if snapshot is valid
        """
        if snapshot_id not in self.snapshot_hashes:
            return False
        
        expected_hash = self.snapshot_hashes[snapshot_id]
        actual_hash = self.compute_snapshot_hash(snapshot_data)
        
        return expected_hash == actual_hash
    
    def register_snapshot(self, snapshot_id: str, snapshot_data: Dict[str, Any]) -> None:
        """
        Register snapshot with integrity hash.
        
        Args:
            snapshot_id: Snapshot identifier
            snapshot_data: Snapshot data
        """
        hash_value = self.compute_snapshot_hash(snapshot_data)
        self.snapshot_hashes[snapshot_id] = hash_value

