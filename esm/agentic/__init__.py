"""
Agentic Cognitive Grammar System for ESM3

This module implements a distributed network of agentic cognitive grammar,
transforming the ESM3 architecture into a recursive, self-modifying system
of cognitive agents.
"""

from .agent import Agent, AgentGrammar, NeuralAgent, SymbolicAgent
from .memory import HypergraphMemory, MemoryNode, MemoryLink
from .task import TaskOrchestrator, TaskDelegator, CognitiveTask
from .attention import AttentionAllocator, ResourceManager
from .autonomy import SelfModifyingKernel, MetaAgent

__all__ = [
    "Agent",
    "AgentGrammar", 
    "HypergraphMemory",
    "MemoryNode",
    "MemoryLink",
    "TaskOrchestrator",
    "TaskDelegator",
    "CognitiveTask",
    "AttentionAllocator",
    "ResourceManager", 
    "SelfModifyingKernel",
    "MetaAgent",
]