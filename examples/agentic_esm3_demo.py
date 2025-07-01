"""
Example demonstrating the Agentic Cognitive Grammar integration with ESM3.

This script shows how to transform the existing ESM3 model into an agentic
architecture while maintaining backward compatibility.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from esm.agentic import (
    NeuralAgent, SymbolicAgent, HypergraphMemory, 
    TaskOrchestrator, CognitiveTask, AttentionAllocator,
    SelfModifyingKernel, MetaAgent
)


class ESM3Agent(NeuralAgent):
    """
    Agentic wrapper for the ESM3 model, transforming it into a cognitive agent
    that can participate in the distributed agentic network.
    """
    
    def __init__(self, esm3_model: Optional[nn.Module] = None, d_model: int = 1280):
        # For demonstration, use a simple mock if no real model provided
        if esm3_model is None:
            esm3_model = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
        
        super().__init__("esm3_protein_agent", esm3_model, d_model)
    
    def _initialize_grammar(self) -> None:
        """Initialize ESM3-specific cognitive grammar."""
        # Inherit base neural grammar
        super()._initialize_grammar()
        
        # Add ESM3-specific cognitive acts
        self.grammar.add_cognitive_act("analyze_sequence", "sequence -> features")
        self.grammar.add_cognitive_act("predict_structure", "sequence -> coordinates")
        self.grammar.add_cognitive_act("annotate_function", "sequence -> annotations")
        self.grammar.add_cognitive_act("fold_protein", "sequence -> structure")
        self.grammar.add_cognitive_act("generate_protein", "constraints -> sequence")
        
        # Update tensor shape for protein analysis
        self.grammar.tensor_shape = (len(self.grammar.cognitive_acts), 5, self.d_model)
    
    def cognitive_act(self, act_name: str, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute ESM3-specific cognitive acts."""
        self.log_cognitive_trail(f"esm3_{act_name}", act_name, input_data)
        
        if act_name in ["analyze_sequence", "predict_structure", "annotate_function"]:
            return self._esm3_inference(input_data, act_name, context)
        elif act_name == "fold_protein":
            return self._fold_protein(input_data, context)
        elif act_name == "generate_protein":
            return self._generate_protein(input_data, context)
        else:
            # Fall back to base neural agent acts
            return super().cognitive_act(act_name, input_data, context)
    
    def _esm3_inference(self, sequence_data: Any, task_type: str, context: Dict[str, Any]) -> torch.Tensor:
        """Perform ESM3 inference for various tasks."""
        # Convert input to tensor if needed
        if isinstance(sequence_data, str):
            # Mock tokenization for demo
            input_tensor = torch.randn(1, len(sequence_data), self.d_model)
        elif torch.is_tensor(sequence_data):
            input_tensor = sequence_data
        else:
            input_tensor = torch.randn(1, 100, self.d_model)
        
        # Run through ESM3 model
        with torch.no_grad():
            output = self.neural_module(input_tensor)
        
        # Store intermediate results in memory if available
        if "memory" in context:
            memory = context["memory"]
            result_node = memory.add_node(
                node_type=f"esm3_{task_type}_result",
                data=output,
                agent_id=self.agent_id
            )
            context["result_node_id"] = result_node
        
        return output
    
    def _fold_protein(self, sequence_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict protein folding structure."""
        # Mock protein folding prediction
        sequence_length = len(sequence_data) if isinstance(sequence_data, str) else 100
        
        folding_result = {
            "coordinates": torch.randn(sequence_length, 3, 3),  # CA, CB, N coordinates
            "confidence": torch.rand(sequence_length),
            "secondary_structure": ["H", "E", "C"] * (sequence_length // 3 + 1)[:sequence_length]
        }
        
        return folding_result
    
    def _generate_protein(self, constraints: Any, context: Dict[str, Any]) -> str:
        """Generate protein sequence based on constraints."""
        # Mock protein generation
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        length = constraints.get("length", 100) if isinstance(constraints, dict) else 100
        
        # Generate random sequence for demo
        import random
        sequence = "".join(random.choice(amino_acids) for _ in range(length))
        
        return sequence


class ProteinAnalysisOrchestrator:
    """
    Orchestrator specifically designed for protein analysis tasks using
    the agentic ESM3 architecture.
    """
    
    def __init__(self):
        # Initialize core agentic infrastructure
        self.memory = HypergraphMemory(embedding_dim=128, max_nodes=5000)
        self.task_orchestrator = TaskOrchestrator(self.memory)
        self.attention_allocator = AttentionAllocator(self.memory, total_attention_budget=1000.0)
        self.kernel = SelfModifyingKernel(self.memory, self.task_orchestrator)
        
        # Create specialized agents for protein analysis
        self.esm3_agent = ESM3Agent()
        self.structure_agent = SymbolicAgent("protein_structure_analyzer")
        self.function_agent = SymbolicAgent("protein_function_analyzer")
        self.meta_agent = MetaAgent(self.kernel)
        
        # Register all agents
        self._register_agents()
        
        # Create protein-specific cognitive acts for symbolic agents
        self._initialize_protein_grammars()
    
    def _register_agents(self):
        """Register all agents with the orchestrator and attention allocator."""
        agents = [self.esm3_agent, self.structure_agent, self.function_agent, self.meta_agent]
        
        for agent in agents:
            self.task_orchestrator.register_agent(agent)
            self.attention_allocator.register_agent(agent)
    
    def _initialize_protein_grammars(self):
        """Initialize protein-specific cognitive grammars for symbolic agents."""
        # Structure analysis grammar
        self.structure_agent.grammar.add_cognitive_act("classify_fold", "structure -> fold_class")
        self.structure_agent.grammar.add_cognitive_act("identify_domains", "structure -> domains")
        self.structure_agent.grammar.add_cognitive_act("analyze_contacts", "structure -> contacts")
        
        # Function analysis grammar  
        self.function_agent.grammar.add_cognitive_act("predict_go_terms", "sequence -> go_terms")
        self.function_agent.grammar.add_cognitive_act("identify_active_sites", "structure -> sites")
        self.function_agent.grammar.add_cognitive_act("predict_interactions", "sequence -> interactions")
    
    def analyze_protein(self, protein_sequence: str) -> Dict[str, Any]:
        """
        Comprehensive protein analysis using the agentic architecture.
        
        This method demonstrates how complex protein analysis tasks are
        decomposed and distributed across the agent network.
        """
        # Create main analysis task
        main_task = CognitiveTask(
            task_type="protein_analysis",
            description=f"Comprehensive analysis of protein sequence: {protein_sequence[:20]}...",
            input_data=protein_sequence,
            required_capabilities=["analyze_sequence"]
        )
        
        # Submit task and let the orchestrator handle decomposition
        task_id = self.task_orchestrator.submit_task(main_task)
        
        # Execute tasks (simplified execution for demo)
        results = {}
        
        # 1. Sequence analysis with ESM3 agent
        seq_result = self.esm3_agent.cognitive_act(
            "analyze_sequence", 
            protein_sequence,
            {"memory": self.memory}
        )
        results["sequence_features"] = seq_result
        
        # 2. Structure prediction
        structure_result = self.esm3_agent.cognitive_act(
            "predict_structure",
            protein_sequence,
            {"memory": self.memory}
        )
        results["predicted_structure"] = structure_result
        
        # 3. Function annotation
        function_result = self.esm3_agent.cognitive_act(
            "annotate_function",
            protein_sequence,
            {"memory": self.memory}
        )
        results["function_annotations"] = function_result
        
        # 4. Update attention based on results
        self._update_attention_based_on_results(results)
        
        # 5. Meta-analysis
        meta_result = self.meta_agent.cognitive_act(
            "observe_system",
            results,
            {"task_id": task_id}
        )
        results["meta_analysis"] = meta_result
        
        return results
    
    def _update_attention_based_on_results(self, results: Dict[str, Any]):
        """Update attention allocation based on analysis results."""
        # Simulate attention updates based on result quality
        attention_updates = {
            self.esm3_agent.agent_id: 10.0,  # High attention for main analysis
            self.structure_agent.agent_id: 5.0,
            self.function_agent.agent_id: 5.0
        }
        
        for node_id, boost in attention_updates.items():
            self.attention_allocator.focus_attention(node_id, boost, "agent")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current state of the agentic system."""
        return {
            "memory_nodes": len(self.memory.nodes),
            "memory_links": len(self.memory.links),
            "active_tasks": len(self.task_orchestrator.active_tasks),
            "registered_agents": len(self.task_orchestrator.agents),
            "attention_distribution": self.attention_allocator.get_high_attention_agents(),
            "evolution_events": len(self.kernel.evolution_history)
        }


def demonstrate_agentic_esm3():
    """Demonstrate the agentic ESM3 system in action."""
    print("🧬 Initializing Agentic ESM3 System...")
    
    # Create the protein analysis orchestrator
    orchestrator = ProteinAnalysisOrchestrator()
    
    print("✅ System initialized with agents:")
    for agent_id, agent in orchestrator.task_orchestrator.agents.items():
        print(f"  - {agent.agent_type} ({agent_id[:8]}...)")
    
    # Example protein sequence (simplified)
    protein_sequence = "MKLLLLLLLLCAAACAGGGGGGGGGGCCCCCCCCHHHHHHHHHHEEEEEEEEEEEEEEEEEE"
    
    print(f"\n🔬 Analyzing protein sequence: {protein_sequence[:30]}...")
    
    # Perform comprehensive analysis
    results = orchestrator.analyze_protein(protein_sequence)
    
    print("\n📊 Analysis Results:")
    print(f"  - Sequence features shape: {results['sequence_features'].shape}")
    print(f"  - Structure prediction shape: {results['predicted_structure'].shape}")
    print(f"  - Function annotations shape: {results['function_annotations'].shape}")
    print(f"  - Meta-analysis: {results['meta_analysis']}")
    
    # Show system state
    print("\n🧠 System State:")
    state = orchestrator.get_system_state()
    for key, value in state.items():
        print(f"  - {key}: {value}")
    
    # Demonstrate cognitive trail introspection
    print("\n🔍 Agent Cognitive Trails:")
    for agent_id, agent in orchestrator.task_orchestrator.agents.items():
        trail_length = len(agent.cognitive_trail)
        if trail_length > 0:
            print(f"  - {agent.agent_type}: {trail_length} cognitive acts")
            latest_act = agent.cognitive_trail[-1]
            print(f"    Latest: {latest_act['intent']} -> {latest_act['action']}")
    
    # Demonstrate attention allocation
    print("\n🎯 Attention Allocation:")
    agents = orchestrator.task_orchestrator.agents
    allocations = orchestrator.attention_allocator.allocate_attention(agents)
    for agent_id, attention in allocations.items():
        agent_type = agents[agent_id].agent_type
        print(f"  - {agent_type}: {attention:.2f}")
    
    print("\n🎉 Agentic ESM3 demonstration complete!")
    
    return orchestrator, results


if __name__ == "__main__":
    # Run the demonstration
    orchestrator, results = demonstrate_agentic_esm3()
    
    print("\n" + "="*60)
    print("AGENTIC COGNITIVE GRAMMAR ARCHITECTURE ACTIVE")
    print("="*60)
    print("The ESM3 model has been transformed into a distributed network")
    print("of cognitive agents capable of recursive task delegation,")
    print("self-modification, and emergent behavior learning.")
    print("="*60)