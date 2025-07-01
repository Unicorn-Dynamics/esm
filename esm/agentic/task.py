"""
Task Orchestration System for agentic cognitive grammar.

Manages task decomposition, delegation, and recursive execution across
the distributed agent network.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set

import torch
import torch.nn as nn

from .agent import Agent
from .memory import HypergraphMemory


class TaskStatus(Enum):
    """Status of a cognitive task."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"


@dataclass
class CognitiveTask:
    """
    Represents a cognitive task that can be executed by agents.
    Tasks can be recursively decomposed into sub-tasks.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "generic"
    description: str = ""
    input_data: Any = None
    expected_output: Any = None
    status: TaskStatus = TaskStatus.PENDING
    priority: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Task hierarchy
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    
    # Agent assignment
    assigned_agent_id: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    
    # Execution tracking
    start_time: float = 0.0
    end_time: float = 0.0
    result: Any = None
    error_message: Optional[str] = None
    
    def add_subtask(self, subtask_id: str) -> None:
        """Add a subtask to this task."""
        self.subtasks.append(subtask_id)
    
    def is_complete(self) -> bool:
        """Check if task is completed."""
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status == TaskStatus.FAILED


class TaskOrchestrator(nn.Module):
    """
    Central orchestrator for managing task execution across agents.
    
    Responsibilities:
    - Task queue management
    - Agent capability matching
    - Task decomposition
    - Result aggregation
    """
    
    def __init__(self, memory: HypergraphMemory):
        super().__init__()
        self.memory = memory
        self.tasks: Dict[str, CognitiveTask] = {}
        self.task_queue: List[str] = []
        self.active_tasks: Set[str] = set()
        self.agents: Dict[str, Agent] = {}
        
        # Task planning neural network
        self.task_planner = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        
        # Store agent capabilities in memory
        capabilities_node = self.memory.add_node(
            node_type="agent_capabilities",
            data=agent.grammar.cognitive_acts,
            agent_id=agent.agent_id
        )
        
        # Link agent to its capabilities
        agent_node = self.memory.add_node(
            node_type="agent",
            data=agent.agent_type,
            agent_id=agent.agent_id
        )
        
        self.memory.add_link(
            link_type="has_capabilities",
            source_nodes=[agent_node],
            target_nodes=[capabilities_node],
            agent_id=agent.agent_id
        )
    
    def submit_task(self, task: CognitiveTask) -> str:
        """Submit a new task for execution."""
        self.tasks[task.task_id] = task
        self.task_queue.append(task.task_id)
        
        # Store task in memory
        task_node = self.memory.add_node(
            node_type="task",
            data=task.description,
            agent_id=None
        )
        task.context["memory_node_id"] = task_node
        
        return task.task_id
    
    def execute_next_task(self) -> Optional[str]:
        """Execute the next task in the queue."""
        if not self.task_queue:
            return None
        
        task_id = self.task_queue.pop(0)
        task = self.tasks[task_id]
        
        # Find suitable agent
        agent = self._find_suitable_agent(task)
        if not agent:
            # Try to decompose task
            self._decompose_task(task)
            return task_id
        
        # Execute task
        return self._execute_task(task, agent)
    
    def _find_suitable_agent(self, task: CognitiveTask) -> Optional[Agent]:
        """Find an agent capable of handling the task."""
        best_agent = None
        best_score = 0.0
        
        for agent in self.agents.values():
            score = self._agent_task_compatibility(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent if best_score > 0.5 else None
    
    def _agent_task_compatibility(self, agent: Agent, task: CognitiveTask) -> float:
        """Compute compatibility score between agent and task."""
        score = 0.0
        
        # Check if agent has required capabilities
        agent_capabilities = set(agent.grammar.cognitive_acts.keys())
        required_capabilities = set(task.required_capabilities)
        
        if required_capabilities.issubset(agent_capabilities):
            score += 1.0
        else:
            # Partial match
            overlap = len(required_capabilities.intersection(agent_capabilities))
            score += overlap / len(required_capabilities) if required_capabilities else 0.0
        
        # Consider agent's current load (simplified)
        score *= 1.0  # Could add load balancing here
        
        return score
    
    def _decompose_task(self, task: CognitiveTask) -> List[str]:
        """Decompose a complex task into smaller subtasks."""
        subtask_ids = []
        
        # Use neural planner to generate subtasks (simplified)
        if task.task_type == "sequence_analysis":
            # Decompose into pattern recognition and structure prediction
            subtasks = [
                CognitiveTask(
                    task_type="pattern_recognition",
                    description=f"Recognize patterns in {task.description}",
                    input_data=task.input_data,
                    parent_task_id=task.task_id,
                    required_capabilities=["pattern_recognize"]
                ),
                CognitiveTask(
                    task_type="structure_prediction", 
                    description=f"Predict structure for {task.description}",
                    input_data=task.input_data,
                    parent_task_id=task.task_id,
                    required_capabilities=["forward"]
                )
            ]
            
            for subtask in subtasks:
                subtask_id = self.submit_task(subtask)
                task.add_subtask(subtask_id)
                subtask_ids.append(subtask_id)
        
        task.status = TaskStatus.DELEGATED
        return subtask_ids
    
    def _execute_task(self, task: CognitiveTask, agent: Agent) -> str:
        """Execute a task using the assigned agent."""
        task.assigned_agent_id = agent.agent_id
        task.status = TaskStatus.RUNNING
        self.active_tasks.add(task.task_id)
        
        try:
            # Determine which cognitive act to use
            cognitive_act = self._select_cognitive_act(task, agent)
            
            # Execute the cognitive act
            result = agent.cognitive_act(
                cognitive_act, 
                task.input_data,
                task.context
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            # Update memory with result
            self._store_task_result(task, result)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
        
        finally:
            self.active_tasks.discard(task.task_id)
        
        return task.task_id
    
    def _select_cognitive_act(self, task: CognitiveTask, agent: Agent) -> str:
        """Select appropriate cognitive act for the task."""
        # Simple mapping based on task type
        act_mapping = {
            "pattern_recognition": "pattern_recognize",
            "structure_prediction": "forward", 
            "inference": "infer",
            "reasoning": "reason"
        }
        
        return act_mapping.get(task.task_type, "forward")
    
    def _store_task_result(self, task: CognitiveTask, result: Any) -> None:
        """Store task result in hypergraph memory."""
        if "memory_node_id" in task.context:
            task_node_id = task.context["memory_node_id"]
            
            # Create result node
            result_node_id = self.memory.add_node(
                node_type="task_result",
                data=result,
                agent_id=task.assigned_agent_id
            )
            
            # Link task to result
            self.memory.add_link(
                link_type="produces",
                source_nodes=[task_node_id],
                target_nodes=[result_node_id],
                agent_id=task.assigned_agent_id
            )


class TaskDelegator:
    """
    Handles recursive task delegation and sub-agent spawning.
    """
    
    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
        self.delegation_history: List[Dict[str, Any]] = []
    
    def delegate_task(
        self, 
        task: CognitiveTask,
        target_agent_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Delegate a task to a specific type of agent."""
        # Find or create suitable agent
        target_agent = self._find_or_create_agent(target_agent_type)
        
        # Create delegation record
        delegation_record = {
            "task_id": task.task_id,
            "delegating_agent": task.assigned_agent_id,
            "target_agent": target_agent.agent_id,
            "context": context or {}
        }
        self.delegation_history.append(delegation_record)
        
        # Execute task with target agent
        return self.orchestrator._execute_task(task, target_agent)
    
    def _find_or_create_agent(self, agent_type: str) -> Agent:
        """Find existing agent of type or create new one."""
        # Look for existing agent of the right type
        for agent in self.orchestrator.agents.values():
            if agent.agent_type == agent_type:
                return agent
        
        # Create new agent (simplified - would need proper factory)
        from .agent import NeuralAgent, SymbolicAgent
        
        if agent_type == "neural":
            new_agent = NeuralAgent(agent_type, nn.Linear(64, 64))
        elif agent_type == "symbolic":
            new_agent = SymbolicAgent(agent_type)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.orchestrator.register_agent(new_agent)
        return new_agent