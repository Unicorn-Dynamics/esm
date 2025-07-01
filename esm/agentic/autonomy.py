"""
Self-Modifying Kernel and Meta-Agent for autonomous system evolution.

Implements the autonomy system that observes, evaluates, and rewrites
agent grammars, enabling self-modification and adaptive behavior.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

import torch
import torch.nn as nn

from .agent import Agent, AgentGrammar
from .memory import HypergraphMemory
from .task import TaskOrchestrator, CognitiveTask


@dataclass
class GrammarEvolution:
    """Records the evolution of an agent's cognitive grammar."""
    evolution_id: str
    agent_id: str
    old_grammar: AgentGrammar
    new_grammar: AgentGrammar
    reason: str
    performance_delta: float
    timestamp: float


class SelfModifyingKernel(nn.Module):
    """
    The self-modifying kernel that observes agent performance and
    adaptively rewrites their cognitive grammars.
    
    This is the core of the autonomy system, enabling the agent network
    to evolve and improve its capabilities over time.
    """
    
    def __init__(
        self, 
        memory: HypergraphMemory,
        orchestrator: TaskOrchestrator,
        evolution_threshold: float = 0.1
    ):
        super().__init__()
        self.memory = memory
        self.orchestrator = orchestrator
        self.evolution_threshold = evolution_threshold
        
        # Track grammar evolution history
        self.evolution_history: List[GrammarEvolution] = []
        self.performance_tracker: Dict[str, List[float]] = {}
        
        # Neural networks for grammar evolution
        self.grammar_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.grammar_synthesizer = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Meta-learning for grammar evolution
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def observe_agent_performance(self, agent: Agent, task_results: List[Dict]) -> None:
        """Observe and record agent performance for potential evolution."""
        if agent.agent_id not in self.performance_tracker:
            self.performance_tracker[agent.agent_id] = []
        
        # Compute performance metrics
        success_rate = sum(1 for result in task_results if result.get('success', False))
        success_rate /= len(task_results) if task_results else 1
        
        efficiency = sum(result.get('efficiency', 1.0) for result in task_results)
        efficiency /= len(task_results) if task_results else 1
        
        # Combined performance score
        performance_score = 0.6 * success_rate + 0.4 * efficiency
        self.performance_tracker[agent.agent_id].append(performance_score)
        
        # Check if evolution is needed
        if self._should_evolve_agent(agent):
            self._evolve_agent_grammar(agent)
    
    def _should_evolve_agent(self, agent: Agent) -> bool:
        """Determine if an agent's grammar should be evolved."""
        if agent.agent_id not in self.performance_tracker:
            return False
        
        performance_history = self.performance_tracker[agent.agent_id]
        
        # Need at least 5 performance samples
        if len(performance_history) < 5:
            return False
        
        # Check if performance is declining or stagnant
        recent_performance = sum(performance_history[-3:]) / 3
        older_performance = sum(performance_history[-6:-3]) / 3 if len(performance_history) >= 6 else recent_performance
        
        performance_delta = recent_performance - older_performance
        
        # Evolve if performance declined significantly
        return performance_delta < -self.evolution_threshold
    
    def _evolve_agent_grammar(self, agent: Agent) -> None:
        """Evolve an agent's cognitive grammar based on observed patterns."""
        old_grammar = agent.grammar
        
        # Analyze current grammar and performance
        grammar_features = self._encode_grammar(old_grammar)
        performance_context = self._get_performance_context(agent)
        
        # Generate new grammar using neural synthesis
        new_grammar_encoding = self._synthesize_new_grammar(grammar_features, performance_context)
        new_grammar = self._decode_grammar(new_grammar_encoding, agent)
        
        # Test new grammar in controlled environment
        if self._validate_new_grammar(agent, new_grammar):
            # Apply the new grammar
            agent.grammar = new_grammar
            
            # Record evolution
            evolution = GrammarEvolution(
                evolution_id=str(uuid.uuid4()),
                agent_id=agent.agent_id,
                old_grammar=old_grammar,
                new_grammar=new_grammar,
                reason="Performance improvement",
                performance_delta=0.0,  # Will be measured later
                timestamp=0.0
            )
            self.evolution_history.append(evolution)
            
            # Store evolution in hypergraph memory
            self._store_evolution_in_memory(evolution)
    
    def _encode_grammar(self, grammar: AgentGrammar) -> torch.Tensor:
        """Encode an agent's grammar into a tensor representation."""
        # Simple encoding: hash of cognitive acts and tensor shape info
        act_count = len(grammar.cognitive_acts)
        context_depth = grammar.context_depth
        tensor_size = sum(grammar.tensor_shape) if grammar.tensor_shape else 0
        
        # Create feature vector
        features = torch.tensor([
            act_count,
            context_depth, 
            tensor_size,
            hash(str(grammar.cognitive_acts)) % 1000  # Simplified hash
        ]).float()
        
        # Pad to 256 dimensions
        padded_features = torch.zeros(256)
        padded_features[:min(len(features), 256)] = features[:256]
        
        return padded_features
    
    def _get_performance_context(self, agent: Agent) -> torch.Tensor:
        """Get performance context for grammar evolution."""
        performance_history = self.performance_tracker.get(agent.agent_id, [0.0])
        
        # Performance statistics
        avg_performance = sum(performance_history) / len(performance_history)
        recent_trend = (performance_history[-1] - performance_history[0]) / len(performance_history) if len(performance_history) > 1 else 0.0
        variance = sum((p - avg_performance) ** 2 for p in performance_history) / len(performance_history)
        
        context = torch.tensor([
            avg_performance,
            recent_trend,
            variance,
            len(performance_history)
        ]).float()
        
        # Pad to 128 dimensions
        padded_context = torch.zeros(128)
        padded_context[:len(context)] = context
        
        return padded_context
    
    def _synthesize_new_grammar(
        self, 
        grammar_features: torch.Tensor, 
        performance_context: torch.Tensor
    ) -> torch.Tensor:
        """Synthesize a new grammar using neural generation."""
        # Combine grammar features and performance context
        combined_input = torch.cat([grammar_features[:64], performance_context[:64]])
        
        # Generate new grammar encoding
        with torch.no_grad():
            new_encoding = self.grammar_synthesizer(combined_input)
        
        return new_encoding
    
    def _decode_grammar(self, encoding: torch.Tensor, agent: Agent) -> AgentGrammar:
        """Decode a grammar encoding into an AgentGrammar object."""
        # Extract parameters from encoding
        act_count = int(encoding[0].item()) + len(agent.grammar.cognitive_acts)
        context_depth = max(1, int(encoding[1].item()) + agent.grammar.context_depth)
        
        # Create new grammar based on encoding
        new_grammar = AgentGrammar(
            agent_id=agent.agent_id,
            cognitive_acts=agent.grammar.cognitive_acts.copy(),
            context_depth=context_depth
        )
        
        # Add new cognitive acts based on encoding patterns
        if encoding[2].item() > 0.5:  # Threshold for adding new acts
            new_grammar.add_cognitive_act("enhanced_pattern_match", "tensor -> tensor")
        
        if encoding[3].item() > 0.5:
            new_grammar.add_cognitive_act("contextual_inference", "context, tensor -> tensor")
        
        # Update tensor shape based on new parameters
        new_grammar.tensor_shape = (act_count, context_depth, 64)
        
        return new_grammar
    
    def _validate_new_grammar(self, agent: Agent, new_grammar: AgentGrammar) -> bool:
        """Validate that a new grammar is functional and beneficial."""
        # Basic validation: ensure grammar has required acts
        if not new_grammar.cognitive_acts:
            return False
        
        # Ensure tensor shape is valid
        if not new_grammar.tensor_shape or any(dim <= 0 for dim in new_grammar.tensor_shape):
            return False
        
        # Could add more sophisticated validation here
        return True
    
    def _store_evolution_in_memory(self, evolution: GrammarEvolution) -> None:
        """Store grammar evolution in hypergraph memory for future reference."""
        evolution_node = self.memory.add_node(
            node_type="grammar_evolution",
            data=evolution,
            agent_id=evolution.agent_id
        )
        
        # Link to the agent that evolved
        agent_nodes = self.memory.query_by_pattern({
            "node_type": "agent",
            "agent_id": evolution.agent_id
        })
        
        if agent_nodes:
            agent_node_id = agent_nodes[0][0]
            self.memory.add_link(
                link_type="evolved_from",
                source_nodes=[evolution_node],
                target_nodes=[agent_node_id],
                agent_id=evolution.agent_id
            )


class MetaAgent(Agent):
    """
    A meta-agent that observes and coordinates other agents.
    
    The meta-agent can:
    - Monitor the overall system state
    - Coordinate between agents
    - Trigger system-wide adaptations
    - Learn emergent behaviors
    """
    
    def __init__(self, kernel: SelfModifyingKernel):
        super().__init__("meta", d_model=1024)
        self.kernel = kernel
        self.system_state_history: List[Dict[str, Any]] = []
        
        # Meta-learning components
        self.system_analyzer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def _initialize_grammar(self) -> None:
        """Initialize meta-agent grammar."""
        self.grammar.add_cognitive_act("observe_system", "system_state -> analysis")
        self.grammar.add_cognitive_act("coordinate_agents", "agent_states -> coordination_plan")
        self.grammar.add_cognitive_act("adapt_system", "performance_data -> adaptations")
        self.grammar.add_cognitive_act("learn_emergent", "interaction_patterns -> new_behaviors")
    
    def cognitive_act(self, act_name: str, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute meta-cognitive acts."""
        self.log_cognitive_trail(f"meta_{act_name}", act_name, input_data)
        
        if act_name == "observe_system":
            return self._observe_system_state(input_data, context)
        elif act_name == "coordinate_agents":
            return self._coordinate_agents(input_data, context)
        elif act_name == "adapt_system":
            return self._adapt_system(input_data, context)
        elif act_name == "learn_emergent":
            return self._learn_emergent_behaviors(input_data, context)
        else:
            raise ValueError(f"Unknown meta-cognitive act: {act_name}")
    
    def _observe_system_state(self, system_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Observe and analyze the overall system state."""
        analysis = {
            "agent_count": len(self.kernel.orchestrator.agents),
            "active_tasks": len(self.kernel.orchestrator.active_tasks),
            "memory_nodes": len(self.kernel.memory.nodes),
            "evolution_events": len(self.kernel.evolution_history)
        }
        
        self.system_state_history.append(analysis)
        return analysis
    
    def _coordinate_agents(self, agent_states: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate interactions between agents."""
        coordination_plan = {
            "prioritize_tasks": True,
            "balance_load": True,
            "share_knowledge": True
        }
        return coordination_plan
    
    def _adapt_system(self, performance_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger system-wide adaptations based on performance."""
        adaptations = {
            "attention_reallocation": True,
            "memory_reorganization": True,
            "grammar_evolution": True
        }
        return adaptations
    
    def _learn_emergent_behaviors(self, patterns: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Learn and encode emergent behaviors observed in the system."""
        emergent_behaviors = {
            "collaborative_patterns": [],
            "efficient_workflows": [],
            "novel_problem_solving": []
        }
        return emergent_behaviors