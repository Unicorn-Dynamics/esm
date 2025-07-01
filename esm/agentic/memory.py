"""
Hypergraph-based Memory System for distributed cognitive agents.

This module implements a distributed semantic store analogous to AtomSpace,
where agents can read/write hypergraph-encoded knowledge and patterns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import networkx as nx


@dataclass
class MemoryNode:
    """
    A node in the hypergraph memory system.
    Represents atomic concepts, patterns, or agent states.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str = "concept"
    data: Any = None
    embedding: Optional[torch.Tensor] = None
    attention_value: float = 0.0
    timestamp: float = 0.0
    agent_id: Optional[str] = None
    
    def __post_init__(self):
        if self.embedding is None:
            # Default embedding dimension
            self.embedding = torch.randn(64)


@dataclass 
class MemoryLink:
    """
    A hyperedge in the memory system connecting multiple nodes.
    Represents relationships, rules, or complex patterns.
    """
    link_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    link_type: str = "relation"
    source_nodes: List[str] = field(default_factory=list)
    target_nodes: List[str] = field(default_factory=list)
    strength: float = 1.0
    attention_value: float = 0.0
    timestamp: float = 0.0
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HypergraphMemory(nn.Module):
    """
    Distributed hypergraph memory system for cognitive agents.
    
    Serves as the central memory substrate where agents can:
    - Store and retrieve patterns
    - Build associative links
    - Share knowledge across the agent network
    - Maintain attention-weighted importance
    """
    
    def __init__(self, embedding_dim: int = 64, max_nodes: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_nodes = max_nodes
        
        # Core data structures
        self.nodes: Dict[str, MemoryNode] = {}
        self.links: Dict[str, MemoryLink] = {}
        self.graph = nx.MultiDiGraph()
        
        # Attention and importance tracking
        self.attention_weights = nn.ParameterDict()
        self.importance_scores = {}
        
        # Embedding networks for dynamic encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.link_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(), 
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def add_node(
        self, 
        node_type: str = "concept",
        data: Any = None,
        agent_id: Optional[str] = None,
        embedding: Optional[torch.Tensor] = None
    ) -> str:
        """Add a new node to the hypergraph memory."""
        if len(self.nodes) >= self.max_nodes:
            self._evict_least_important_node()
        
        node = MemoryNode(
            node_type=node_type,
            data=data,
            agent_id=agent_id,
            embedding=embedding
        )
        
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **{
            "type": node_type,
            "agent_id": agent_id,
            "attention": 0.0
        })
        
        # Initialize attention weight parameter
        self.attention_weights[node.node_id] = nn.Parameter(torch.tensor(0.0))
        
        return node.node_id
    
    def add_link(
        self,
        link_type: str,
        source_nodes: List[str],
        target_nodes: List[str],
        strength: float = 1.0,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a hyperedge linking multiple nodes."""
        link = MemoryLink(
            link_type=link_type,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            strength=strength,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        self.links[link.link_id] = link
        
        # Add edges to graph for all source->target pairs
        for source in source_nodes:
            for target in target_nodes:
                self.graph.add_edge(source, target, 
                                  link_id=link.link_id,
                                  link_type=link_type,
                                  strength=strength)
        
        return link.link_id
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_link(self, link_id: str) -> Optional[MemoryLink]:
        """Retrieve a link by its ID."""
        return self.links.get(link_id)
    
    def query_by_pattern(
        self, 
        pattern: Dict[str, Any],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Query nodes by pattern matching and return top-k results.
        Returns list of (node_id, similarity_score) tuples.
        """
        if "embedding" in pattern:
            query_embedding = pattern["embedding"]
            results = []
            
            for node_id, node in self.nodes.items():
                if node.embedding is not None:
                    similarity = torch.cosine_similarity(
                        query_embedding.unsqueeze(0),
                        node.embedding.unsqueeze(0)
                    ).item()
                    results.append((node_id, similarity))
            
            # Sort by similarity and return top-k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
        
        # Pattern matching on node attributes
        results = []
        for node_id, node in self.nodes.items():
            score = self._pattern_match_score(node, pattern)
            if score > 0:
                results.append((node_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def get_neighbors(
        self, 
        node_id: str, 
        hops: int = 1,
        link_types: Optional[List[str]] = None
    ) -> List[str]:
        """Get neighboring nodes within specified hops."""
        if node_id not in self.graph:
            return []
        
        neighbors = set()
        current_nodes = {node_id}
        
        for _ in range(hops):
            next_nodes = set()
            for node in current_nodes:
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    if link_types is None or any(
                        data.get("link_type") in link_types 
                        for data in edge_data.values()
                    ):
                        next_nodes.add(neighbor)
            neighbors.update(next_nodes)
            current_nodes = next_nodes
        
        return list(neighbors)
    
    def update_attention(self, node_id: str, attention_delta: float) -> None:
        """Update attention value for a node (ECAN-style)."""
        if node_id in self.nodes:
            self.nodes[node_id].attention_value += attention_delta
            if node_id in self.attention_weights:
                self.attention_weights[node_id].data += attention_delta
    
    def get_high_attention_nodes(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get nodes with highest attention values."""
        attention_scores = [
            (node_id, node.attention_value)
            for node_id, node in self.nodes.items()
        ]
        attention_scores.sort(key=lambda x: x[1], reverse=True)
        return attention_scores[:k]
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Neural forward pass for memory encoding/retrieval."""
        # Encode query
        encoded_query = self.node_encoder(query)
        
        # Simple attention mechanism over all node embeddings
        node_embeddings = torch.stack([
            node.embedding for node in self.nodes.values() 
            if node.embedding is not None
        ]) if self.nodes else torch.zeros(1, self.embedding_dim)
        
        if node_embeddings.shape[0] == 0:
            return encoded_query
        
        # Compute attention weights
        attention_scores = torch.matmul(encoded_query, node_embeddings.T)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Weighted sum of node embeddings
        retrieved_memory = torch.matmul(attention_weights, node_embeddings)
        
        return retrieved_memory
    
    def _pattern_match_score(self, node: MemoryNode, pattern: Dict[str, Any]) -> float:
        """Compute pattern matching score for a node."""
        score = 0.0
        
        if "node_type" in pattern and node.node_type == pattern["node_type"]:
            score += 1.0
        
        if "agent_id" in pattern and node.agent_id == pattern["agent_id"]:
            score += 1.0
        
        # Add more pattern matching logic as needed
        return score
    
    def _evict_least_important_node(self) -> None:
        """Evict the least important node when memory is full."""
        if not self.nodes:
            return
        
        # Find node with lowest attention value
        least_important = min(
            self.nodes.items(),
            key=lambda x: x[1].attention_value
        )
        
        node_id = least_important[0]
        
        # Remove from all data structures
        del self.nodes[node_id]
        if node_id in self.attention_weights:
            del self.attention_weights[node_id]
        if node_id in self.graph:
            self.graph.remove_node(node_id)
        
        # Remove any links that reference this node
        links_to_remove = []
        for link_id, link in self.links.items():
            if node_id in link.source_nodes or node_id in link.target_nodes:
                links_to_remove.append(link_id)
        
        for link_id in links_to_remove:
            del self.links[link_id]