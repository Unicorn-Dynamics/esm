"""
Attention Allocation System (ECAN-inspired) for cognitive agents.

Manages attention allocation, resource optimization, and priority 
assignment across the distributed agent network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .agent import Agent
from .memory import HypergraphMemory


@dataclass
class AttentionState:
    """Represents the attention state of an agent or memory node."""
    short_term_importance: float = 0.0
    long_term_importance: float = 0.0
    urgency: float = 0.0
    confidence: float = 1.0
    
    def total_attention(self) -> float:
        """Compute total attention value."""
        return (self.short_term_importance + 
                self.long_term_importance + 
                self.urgency) * self.confidence


class AttentionAllocator(nn.Module):
    """
    ECAN-inspired attention allocation mechanism for cognitive agents.
    
    Manages the flow of attention across:
    - Active agents and their cognitive processes
    - Memory nodes and patterns in the hypergraph
    - Tasks and their priority levels
    """
    
    def __init__(
        self, 
        memory: HypergraphMemory,
        total_attention_budget: float = 100.0,
        decay_rate: float = 0.95
    ):
        super().__init__()
        self.memory = memory
        self.total_attention_budget = total_attention_budget
        self.decay_rate = decay_rate
        
        # Attention states for agents and memory nodes
        self.agent_attention: Dict[str, AttentionState] = {}
        self.memory_attention: Dict[str, AttentionState] = {}
        
        # Neural networks for attention computation
        self.attention_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # [STI, LTI, urgency, confidence]
        )
        
        # Importance spreading network
        self.importance_spreader = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent for attention management."""
        self.agent_attention[agent.agent_id] = AttentionState()
    
    def allocate_attention(self, agents: Dict[str, Agent]) -> Dict[str, float]:
        """
        Allocate attention budget across active agents.
        Returns mapping of agent_id -> attention_amount.
        """
        if not agents:
            return {}
        
        # Compute attention demands for each agent
        attention_demands = {}
        total_demand = 0.0
        
        for agent_id, agent in agents.items():
            demand = self._compute_attention_demand(agent)
            attention_demands[agent_id] = demand
            total_demand += demand
        
        # Normalize and allocate based on budget
        allocations = {}
        if total_demand > 0:
            for agent_id, demand in attention_demands.items():
                allocation = (demand / total_demand) * self.total_attention_budget
                allocations[agent_id] = allocation
                
                # Update agent's attention state
                if agent_id in self.agent_attention:
                    self.agent_attention[agent_id].short_term_importance = allocation
        
        return allocations
    
    def update_memory_attention(self, node_interactions: Dict[str, float]) -> None:
        """Update attention values for memory nodes based on interactions."""
        for node_id, interaction_strength in node_interactions.items():
            if node_id in self.memory_attention:
                attention_state = self.memory_attention[node_id]
                
                # Increase short-term importance based on interaction
                attention_state.short_term_importance += interaction_strength
                
                # Spread importance to neighboring nodes
                self._spread_importance(node_id, interaction_strength * 0.5)
            else:
                # Create new attention state for this node
                self.memory_attention[node_id] = AttentionState(
                    short_term_importance=interaction_strength
                )
        
        # Apply attention decay
        self._apply_attention_decay()
    
    def get_high_attention_agents(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get agents with highest attention values."""
        agent_scores = [
            (agent_id, state.total_attention())
            for agent_id, state in self.agent_attention.items()
        ]
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores[:k]
    
    def get_high_attention_memory(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get memory nodes with highest attention values."""
        memory_scores = [
            (node_id, state.total_attention())
            for node_id, state in self.memory_attention.items()
        ]
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        return memory_scores[:k]
    
    def focus_attention(
        self, 
        target_id: str, 
        attention_boost: float,
        target_type: str = "agent"
    ) -> None:
        """Manually focus additional attention on a specific target."""
        if target_type == "agent" and target_id in self.agent_attention:
            self.agent_attention[target_id].urgency += attention_boost
        elif target_type == "memory" and target_id in self.memory_attention:
            self.memory_attention[target_id].urgency += attention_boost
    
    def _compute_attention_demand(self, agent: Agent) -> float:
        """Compute how much attention an agent currently demands."""
        # Base demand on agent's current state and activity
        state_tensor = agent.get_state_tensor()
        
        # Use neural network to predict attention demand
        with torch.no_grad():
            # Create feature vector (simplified)
            remaining_dims = max(0, 128 - state_tensor.shape[0] - 2)
            features = torch.cat([
                state_tensor,
                torch.tensor([len(agent.cognitive_trail)]).float(),
                torch.tensor([agent.grammar.context_depth]).float(),
                torch.zeros(remaining_dims)  # Pad to 128
            ])[:128]
            
            attention_values = self.attention_predictor(features)
            sti, lti, urgency, confidence = attention_values
            
            attention_state = AttentionState(
                short_term_importance=sti.item(),
                long_term_importance=lti.item(), 
                urgency=urgency.item(),
                confidence=torch.sigmoid(confidence).item()
            )
            
            self.agent_attention[agent.agent_id] = attention_state
            return attention_state.total_attention()
    
    def _spread_importance(self, source_node_id: str, importance_amount: float) -> None:
        """Spread importance from a node to its neighbors (ECAN-style)."""
        neighbors = self.memory.get_neighbors(source_node_id, hops=1)
        
        if not neighbors:
            return
        
        # Distribute importance equally among neighbors (simplified)
        importance_per_neighbor = importance_amount / len(neighbors)
        
        for neighbor_id in neighbors:
            if neighbor_id in self.memory_attention:
                self.memory_attention[neighbor_id].short_term_importance += importance_per_neighbor
            else:
                self.memory_attention[neighbor_id] = AttentionState(
                    short_term_importance=importance_per_neighbor
                )
    
    def _apply_attention_decay(self) -> None:
        """Apply exponential decay to attention values."""
        for attention_state in self.agent_attention.values():
            attention_state.short_term_importance *= self.decay_rate
            attention_state.urgency *= self.decay_rate
        
        for attention_state in self.memory_attention.values():
            attention_state.short_term_importance *= self.decay_rate
            attention_state.urgency *= self.decay_rate


class ResourceManager:
    """
    Manages computational resources across the agent network.
    """
    
    def __init__(self, total_compute_budget: float = 1000.0):
        self.total_compute_budget = total_compute_budget
        self.current_allocations: Dict[str, float] = {}
        self.resource_history: List[Dict[str, float]] = []
    
    def allocate_resources(
        self, 
        attention_allocations: Dict[str, float],
        agent_priorities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Allocate computational resources based on attention and priorities.
        """
        total_attention = sum(attention_allocations.values())
        resource_allocations = {}
        
        if total_attention > 0:
            for agent_id, attention in attention_allocations.items():
                priority_weight = agent_priorities.get(agent_id, 1.0)
                
                # Compute resource allocation
                attention_ratio = attention / total_attention
                resource_amount = attention_ratio * priority_weight * self.total_compute_budget
                
                resource_allocations[agent_id] = resource_amount
        
        self.current_allocations = resource_allocations
        self.resource_history.append(resource_allocations.copy())
        
        return resource_allocations
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization statistics."""
        if not self.current_allocations:
            return {}
        
        total_allocated = sum(self.current_allocations.values())
        
        return {
            "total_allocated": total_allocated,
            "utilization_rate": total_allocated / self.total_compute_budget,
            "free_resources": self.total_compute_budget - total_allocated,
            "agent_count": len(self.current_allocations)
        }
    
    def optimize_allocation(self) -> Dict[str, float]:
        """
        Optimize resource allocation based on historical performance.
        """
        # Simple optimization: redistribute resources more evenly
        if not self.current_allocations:
            return {}
        
        agent_count = len(self.current_allocations)
        base_allocation = self.total_compute_budget / agent_count
        
        optimized_allocations = {}
        for agent_id in self.current_allocations:
            current = self.current_allocations[agent_id]
            # Blend current allocation with equal distribution
            optimized = 0.7 * current + 0.3 * base_allocation
            optimized_allocations[agent_id] = optimized
        
        return optimized_allocations