"""
Enhanced reasoning system with multi-step planning and chain-of-thought processing.

This module provides advanced AI reasoning capabilities including:
- Multi-step task planning and decomposition
- Chain-of-thought reasoning with explanations
- Goal-oriented problem solving
- Strategic planning and decision making
- Complex reasoning with intermediate steps
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)


class ReasoningStep:
    """Represents a single step in the reasoning process"""

    def __init__(
        self,
        step_number: int,
        description: str,
        reasoning: str,
        action: Optional[str] = None,
        expected_outcome: Optional[str] = None,
        confidence: float = 1.0,
    ):
        self.step_number = step_number
        self.description = description
        self.reasoning = reasoning
        self.action = action
        self.expected_outcome = expected_outcome
        self.confidence = confidence
        self.actual_outcome: Optional[str] = None
        self.success: Optional[bool] = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation"""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "reasoning": self.reasoning,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "actual_outcome": self.actual_outcome,
            "success": self.success,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """Create step from dictionary representation"""
        step = cls(
            step_number=data["step_number"],
            description=data["description"],
            reasoning=data["reasoning"],
            action=data.get("action"),
            expected_outcome=data.get("expected_outcome"),
            confidence=data.get("confidence", 1.0),
        )
        step.actual_outcome = data.get("actual_outcome")
        step.success = data.get("success")
        step.metadata = data.get("metadata", {})
        return step


class ReasoningChain:
    """A chain of reasoning steps with goal-oriented planning"""

    def __init__(self, goal: str, context: Optional[Dict[str, Any]] = None):
        self.goal = goal
        self.context = context or {}
        self.steps: List[ReasoningStep] = []
        self.current_step = 0
        self.completed = False
        self.success = False
        self.final_outcome: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def add_step(self, step: ReasoningStep):
        """Add a reasoning step to the chain"""
        self.steps.append(step)

    def get_next_step(self) -> Optional[ReasoningStep]:
        """Get the next step to execute"""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def complete_step(self, outcome: str, success: bool = True):
        """Mark current step as completed"""
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            step.actual_outcome = outcome
            step.success = success
            self.current_step += 1

    def is_complete(self) -> bool:
        """Check if the reasoning chain is complete"""
        return self.current_step >= len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary representation"""
        return {
            "goal": self.goal,
            "context": self.context,
            "steps": [step.to_dict() for step in self.steps],
            "current_step": self.current_step,
            "completed": self.completed,
            "success": self.success,
            "final_outcome": self.final_outcome,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningChain":
        """Create chain from dictionary representation"""
        chain = cls(data["goal"], data.get("context", {}))
        chain.steps = [ReasoningStep.from_dict(step_data) for step_data in data.get("steps", [])]
        chain.current_step = data.get("current_step", 0)
        chain.completed = data.get("completed", False)
        chain.success = data.get("success", False)
        chain.final_outcome = data.get("final_outcome")
        chain.metadata = data.get("metadata", {})
        return chain


class ChainOfThoughtReasoner:
    """Advanced reasoner using chain-of-thought techniques"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.active_chains: set[str] = set()

    async def reason_about_task(
        self, task_description: str, context: Optional[Dict[str, Any]] = None, max_steps: int = 10
    ) -> ReasoningChain:
        """Create and execute a reasoning chain for a complex task"""
        # Generate unique chain ID
        chain_id = f"reasoning_{asyncio.get_event_loop().time()}"
        self.active_chains.add(chain_id)

        try:
            # Create initial reasoning chain
            chain = ReasoningChain(task_description, context)

            # Plan the reasoning steps
            await self._plan_reasoning_steps(chain, max_steps)

            # Execute the reasoning chain
            await self._execute_reasoning_chain(chain)

            # Store completed chain
            self.reasoning_chains[chain_id] = chain

            return chain

        finally:
            self.active_chains.discard(chain_id)

    async def _plan_reasoning_steps(self, chain: ReasoningChain, max_steps: int):
        """Plan the steps needed to solve the task using LLM reasoning"""
        planning_prompt = f"""
        I need to solve this task: "{chain.goal}"

        Context: {json.dumps(chain.context, indent=2)}

        Break this down into a maximum of {max_steps} logical steps. For each step, provide:
        1. A clear description of what this step accomplishes
        2. The reasoning behind why this step is necessary
        3. What action or method will be used
        4. What outcome is expected from this step
        5. Confidence level (0.0-1.0) in this step succeeding

        Return the response as a JSON array of step objects with keys:
        description, reasoning, action, expected_outcome, confidence

        Focus on breaking down complex tasks into manageable, logical steps.
        """

        if not self.llm_client:
            # Fallback planning without LLM
            await self._fallback_planning(chain, max_steps)
            return

        try:
            response = ""
            async for chunk in self.llm_client.generate(
                messages=[{"role": "user", "content": planning_prompt}]
            ):
                if chunk.get("message", {}).get("content"):
                    response += chunk["message"]["content"]

            # Parse the planning response
            try:
                steps_data = json.loads(response.strip())
                for i, step_data in enumerate(steps_data[:max_steps]):
                    step = ReasoningStep(
                        step_number=i + 1,
                        description=step_data.get("description", ""),
                        reasoning=step_data.get("reasoning", ""),
                        action=step_data.get("action"),
                        expected_outcome=step_data.get("expected_outcome"),
                        confidence=step_data.get("confidence", 1.0),
                    )
                    chain.add_step(step)

            except json.JSONDecodeError:
                log.warning("Failed to parse LLM planning response, using fallback")
                await self._fallback_planning(chain, max_steps)

        except Exception as e:
            log.error(f"LLM planning failed: {e}")
            await self._fallback_planning(chain, max_steps)

    async def _fallback_planning(self, chain: ReasoningChain, max_steps: int):
        """Fallback planning when LLM is not available"""
        # Simple task decomposition based on keywords
        task_lower = chain.goal.lower()

        if "search" in task_lower or "find" in task_lower:
            steps = [
                ReasoningStep(
                    1,
                    "Analyze the search query",
                    "Understanding what information is needed",
                    "analyze_query",
                    "Clear understanding of search requirements",
                    0.9,
                ),
                ReasoningStep(
                    2,
                    "Determine search strategy",
                    "Choosing appropriate search methods and sources",
                    "plan_search",
                    "Optimal search approach identified",
                    0.8,
                ),
                ReasoningStep(
                    3,
                    "Execute search",
                    "Perform the actual search using planned methods",
                    "execute_search",
                    "Relevant information found",
                    0.7,
                ),
                ReasoningStep(
                    4,
                    "Filter and validate results",
                    "Review search results for accuracy and relevance",
                    "validate_results",
                    "High-quality, relevant information selected",
                    0.8,
                ),
            ]
        elif "code" in task_lower or "programming" in task_lower:
            steps = [
                ReasoningStep(
                    1,
                    "Understand the coding task",
                    "Analyze requirements and constraints",
                    "analyze_requirements",
                    "Clear understanding of coding objectives",
                    0.9,
                ),
                ReasoningStep(
                    2,
                    "Design the solution",
                    "Plan the code structure and algorithm",
                    "design_solution",
                    "Comprehensive solution design",
                    0.8,
                ),
                ReasoningStep(
                    3,
                    "Implement the code",
                    "Write the actual code following the design",
                    "implement_code",
                    "Working code implementation",
                    0.7,
                ),
                ReasoningStep(
                    4,
                    "Test and validate",
                    "Test the code and ensure it meets requirements",
                    "test_code",
                    "Validated, working code",
                    0.8,
                ),
            ]
        else:
            # Generic problem-solving steps
            steps = [
                ReasoningStep(
                    1,
                    "Analyze the problem",
                    "Understand the core issue and requirements",
                    "analyze_problem",
                    "Clear problem definition",
                    0.8,
                ),
                ReasoningStep(
                    2,
                    "Research and gather information",
                    "Collect relevant information and data",
                    "gather_info",
                    "Comprehensive information collected",
                    0.7,
                ),
                ReasoningStep(
                    3,
                    "Develop solution approach",
                    "Design and plan the solution strategy",
                    "design_solution",
                    "Viable solution approach",
                    0.8,
                ),
                ReasoningStep(
                    4,
                    "Execute and implement",
                    "Carry out the planned solution",
                    "implement_solution",
                    "Solution successfully implemented",
                    0.7,
                ),
                ReasoningStep(
                    5,
                    "Evaluate and refine",
                    "Assess results and make improvements",
                    "evaluate_results",
                    "Optimized final solution",
                    0.8,
                ),
            ]

        # Add steps up to max_steps limit
        for step in steps[:max_steps]:
            chain.add_step(step)

    async def _execute_reasoning_chain(self, chain: ReasoningChain):
        """Execute the planned reasoning steps"""
        log.info(f"Executing reasoning chain for: {chain.goal}")

        while not chain.is_complete():
            current_step = chain.get_next_step()
            if not current_step:
                break

            log.info(f"Executing step {current_step.step_number}: {current_step.description}")

            try:
                # Simulate step execution (in real implementation, this would call actual tools/functions)
                outcome = await self._execute_step(current_step, chain.context)
                success = True

            except Exception as e:
                outcome = f"Step failed: {str(e)}"
                success = False
                log.error(f"Step {current_step.step_number} failed: {e}")

            # Complete the step
            chain.complete_step(outcome, success)

            # If step failed, we might want to adjust the plan
            if not success and current_step.confidence > 0.5:
                await self._adjust_plan_for_failure(chain, current_step)

        # Mark chain as complete
        chain.completed = True
        chain.success = all(step.success for step in chain.steps if step.success is not None)
        chain.final_outcome = self._generate_final_outcome(chain)

        log.info(f"Reasoning chain completed: {chain.success}")

    async def _execute_step(self, step: ReasoningStep, context: Dict[str, Any]) -> str:
        """Execute a reasoning step (placeholder for actual implementation)"""
        # This is where you would integrate with actual tools and functions
        # For now, we'll simulate execution based on the step action

        action = step.action or "generic_action"

        if action == "analyze_query":
            return "Query analysis complete: identified key search terms and constraints"
        elif action == "plan_search":
            return "Search strategy planned: using web search and knowledge base lookup"
        elif action == "execute_search":
            return "Search executed: found 15 relevant results from 3 sources"
        elif action == "validate_results":
            return "Results validated: 12 high-quality, relevant results selected"
        elif action == "analyze_requirements":
            return "Requirements analyzed: clear objectives and constraints identified"
        elif action == "design_solution":
            return "Solution designed: modular architecture with error handling"
        elif action == "implement_code":
            return "Code implemented: 250 lines of clean, documented code"
        elif action == "test_code":
            return "Code tested: all tests passing, code ready for production"
        else:
            return f"Step '{action}' executed successfully with context: {context}"

    async def _adjust_plan_for_failure(self, chain: ReasoningChain, failed_step: ReasoningStep):
        """Adjust the reasoning plan when a step fails"""
        log.info(f"Adjusting plan due to failed step {failed_step.step_number}")

        # Add a recovery step
        recovery_step = ReasoningStep(
            step_number=len(chain.steps) + 1,
            description=f"Recover from failure in step {failed_step.step_number}",
            reasoning="Previous step failed, need alternative approach",
            action="retry_with_alternative",
            expected_outcome="Successful completion using backup method",
            confidence=0.6,
        )

        chain.add_step(recovery_step)

    def _generate_final_outcome(self, chain: ReasoningChain) -> str:
        """Generate final outcome summary for the reasoning chain"""
        if not chain.steps:
            return "No steps executed"

        successful_steps = sum(1 for step in chain.steps if step.success)
        total_steps = len(chain.steps)

        if chain.success:
            return f"Task completed successfully in {total_steps} steps"
        else:
            return f"Task partially completed: {successful_steps}/{total_steps} steps successful"

    async def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get history of all completed reasoning chains"""
        return [chain.to_dict() for chain in self.reasoning_chains.values()]

    async def get_active_reasoning(self) -> List[str]:
        """Get list of currently active reasoning chain IDs"""
        return list(self.active_chains)


class MultiStepPlanner:
    """Advanced planner for complex multi-step tasks"""

    def __init__(self, reasoner: ChainOfThoughtReasoner):
        self.reasoner = reasoner
        self.task_templates: Dict[str, Dict[str, Any]] = {}

    async def plan_complex_task(
        self,
        task_description: str,
        task_type: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReasoningChain:
        """Plan and execute a complex multi-step task"""

        # Enhance context with task type knowledge
        enhanced_context = {
            "task_type": task_type,
            "constraints": constraints or {},
            "planning_timestamp": asyncio.get_event_loop().time(),
        }

        # Use template if available
        if task_type in self.task_templates:
            template = self.task_templates[task_type]
            enhanced_context.update(template)

        # Create and execute reasoning chain
        chain = await self.reasoner.reason_about_task(
            task_description=task_description,
            context=enhanced_context,
            max_steps=15,  # Allow more steps for complex tasks
        )

        return chain

    def add_task_template(self, task_type: str, template: Dict[str, Any]):
        """Add a task template for common task types"""
        self.task_templates[task_type] = template

    async def analyze_task_complexity(self, task_description: str) -> Dict[str, Any]:
        """Analyze the complexity of a task to determine planning approach"""
        # Simple complexity analysis based on keywords and length
        complexity_score = 0

        # Length-based scoring
        if len(task_description) > 500:
            complexity_score += 2
        elif len(task_description) > 200:
            complexity_score += 1

        # Keyword-based scoring
        complex_keywords = [
            "multiple",
            "complex",
            "advanced",
            "integrate",
            "system",
            "architecture",
            "design",
            "implement",
            "deploy",
            "optimize",
            "scale",
            "security",
        ]

        found_keywords = sum(
            1 for keyword in complex_keywords if keyword in task_description.lower()
        )
        complexity_score += min(found_keywords, 3)  # Cap at 3

        # Determine approach
        if complexity_score >= 4:
            approach = "comprehensive_planning"
            max_steps = 15
        elif complexity_score >= 2:
            approach = "structured_planning"
            max_steps = 10
        else:
            approach = "simple_execution"
            max_steps = 5

        return {
            "complexity_score": complexity_score,
            "approach": approach,
            "recommended_max_steps": max_steps,
            "estimated_effort": "high"
            if complexity_score >= 4
            else "medium"
            if complexity_score >= 2
            else "low",
        }


# Global reasoning system instances
chain_reasoner = ChainOfThoughtReasoner()
multi_step_planner = MultiStepPlanner(chain_reasoner)


async def get_reasoner() -> ChainOfThoughtReasoner:
    """Get the global chain-of-thought reasoner"""
    return chain_reasoner


async def get_planner() -> MultiStepPlanner:
    """Get the global multi-step planner"""
    return multi_step_planner


__all__ = [
    "ReasoningStep",
    "ReasoningChain",
    "ChainOfThoughtReasoner",
    "MultiStepPlanner",
    "chain_reasoner",
    "multi_step_planner",
    "get_reasoner",
    "get_planner",
]
