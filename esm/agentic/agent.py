"""
Base Agent class for the Agentic Cognitive Grammar system.

Each agent exposes a cognitive grammar (API) for cognitive acts and can
participate in the distributed hypergraph of neural-symbolic emergence.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import torch
import torch.nn as nn


@dataclass
class AgentGrammar:
    """
    Defines the cognitive grammar (API) that an agent exposes.
    Each grammar consists of cognitive acts (methods) and their signatures.
    """
    agent_id: str
    cognitive_acts: Dict[str, str] = field(default_factory=dict)
    tensor_shape: tuple = field(default_factory=tuple)
    context_depth: int = 1
    
    def add_cognitive_act(self, name: str, signature: str) -> None:
        """Register a new cognitive act in this agent's grammar."""
        self.cognitive_acts[name] = signature
    
    def get_tensor_representation(self) -> torch.Tensor:
        """Get tensor representation of this agent's state."""
        if not self.tensor_shape:
            # Default tensor shape based on action degrees and context depth
            self.tensor_shape = (len(self.cognitive_acts), self.context_depth, 64)
        return torch.zeros(self.tensor_shape)


class CognitiveAct(Protocol):
    """Protocol for cognitive acts that agents can perform."""
    
    def __call__(self, input_data: Any, context: Dict[str, Any]) -> Any:
        ...


class Agent(nn.Module, ABC):
    """
    Base class for all cognitive agents in the distributed network.
    
    Each agent:
    - Exposes a cognitive grammar (API) of cognitive acts
    - Maintains its state as a tensor field
    - Can send/receive hypergraph-encoded messages  
    - Logs its cognitive trail for self-reflection
    """
    
    def __init__(self, agent_type: str, d_model: int = 512):
        super().__init__()
        self.agent_id = str(uuid.uuid4())
        self.agent_type = agent_type
        self.d_model = d_model
        self.grammar = AgentGrammar(self.agent_id)
        self.cognitive_trail: List[Dict[str, Any]] = []
        
        # Tensor state representation
        self.state_tensor = nn.Parameter(torch.randn(d_model))
        
        # Initialize agent-specific grammar
        self._initialize_grammar()
    
    @abstractmethod
    def _initialize_grammar(self) -> None:
        """Initialize the cognitive grammar for this agent type."""
        pass
    
    @abstractmethod
    def cognitive_act(self, act_name: str, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute a cognitive act defined in this agent's grammar."""
        pass
    
    def log_cognitive_trail(self, intent: str, action: str, result: Any) -> None:
        """Log a cognitive act for self-reflection and meta-learning."""
        trail_entry = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "intent": intent,
            "action": action,
            "result": str(result)[:100],  # Truncate for storage
            "timestamp": torch.tensor(0.0),  # In real implementation, use actual timestamp
            "state_snapshot": self.state_tensor.detach().clone()
        }
        self.cognitive_trail.append(trail_entry)
    
    def get_state_tensor(self) -> torch.Tensor:
        """Get current tensor representation of agent state."""
        return self.state_tensor
    
    def update_state(self, new_state: torch.Tensor) -> None:
        """Update agent's tensor state."""
        self.state_tensor.data = new_state.data
    
    def introspect(self) -> Dict[str, Any]:
        """Self-reflective introspection of cognitive trail."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "grammar": self.grammar,
            "trail_length": len(self.cognitive_trail),
            "state_norm": self.state_tensor.norm().item()
        }


class NeuralAgent(Agent):
    """Agent that wraps neural network functionality."""
    
    def __init__(self, agent_type: str, neural_module: nn.Module, d_model: int = 512):
        super().__init__(agent_type, d_model)
        self.neural_module = neural_module
    
    def _initialize_grammar(self) -> None:
        """Initialize grammar for neural operations."""
        self.grammar.add_cognitive_act("forward", "tensor -> tensor")
        self.grammar.add_cognitive_act("backward", "tensor -> None")
        self.grammar.add_cognitive_act("pattern_recognize", "tensor -> tensor")
    
    def cognitive_act(self, act_name: str, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute neural cognitive acts."""
        self.log_cognitive_trail(f"execute_{act_name}", act_name, input_data)
        
        if act_name == "forward":
            return self.neural_module(input_data)
        elif act_name == "pattern_recognize":
            # Extract pattern features
            return self.neural_module(input_data)
        else:
            raise ValueError(f"Unknown cognitive act: {act_name}")


class SymbolicAgent(Agent):
    """Agent that performs symbolic reasoning operations."""
    
    def __init__(self, agent_type: str = "symbolic", d_model: int = 512):
        super().__init__(agent_type, d_model)
        self.rules: Dict[str, Any] = {}
    
    def _initialize_grammar(self) -> None:
        """Initialize grammar for symbolic operations."""
        self.grammar.add_cognitive_act("infer", "facts -> conclusions")
        self.grammar.add_cognitive_act("reason", "premises -> result")
        self.grammar.add_cognitive_act("compose", "patterns -> pattern")
    
    def cognitive_act(self, act_name: str, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute symbolic cognitive acts."""
        self.log_cognitive_trail(f"symbolic_{act_name}", act_name, input_data)
        
        if act_name == "infer":
            # Apply inference rules
            return self._apply_inference(input_data, context)
        elif act_name == "reason":
            # Logical reasoning
            return self._logical_reasoning(input_data, context)
        elif act_name == "compose":
            # Pattern composition
            return self._compose_patterns(input_data, context)
        else:
            raise ValueError(f"Unknown cognitive act: {act_name}")
    
    def _apply_inference(self, facts: Any, context: Dict[str, Any]) -> Any:
        """Apply symbolic inference rules."""
        # Placeholder for symbolic inference
        return facts
    
    def _logical_reasoning(self, premises: Any, context: Dict[str, Any]) -> Any:
        """Perform logical reasoning."""
        # Placeholder for logical reasoning
        return premises
    
    def _compose_patterns(self, patterns: Any, context: Dict[str, Any]) -> Any:
        """Compose multiple patterns into new patterns."""
        # Placeholder for pattern composition
        return patterns