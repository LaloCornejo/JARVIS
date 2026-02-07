"""Enhanced tools for JARVIS - Agents, Workflows, Users, and Memory"""

from __future__ import annotations

import logging

from tools.base import BaseTool, ToolResult

log = logging.getLogger("jarvis.tools.enhanced")


class AgentExecuteTool(BaseTool):
    """Execute a request through a specialized agent"""

    name = "agent_execute"
    description = (
        "Execute a request using a specialized AI agent (code_review, research, creative, planning)"
    )
    parameters = {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["code_review", "research", "creative", "planning"],
                "description": "Type of specialized agent to use",
            },
            "request": {"type": "string", "description": "The request or task for the agent"},
            "context": {
                "type": "object",
                "description": "Additional context for the agent",
                "default": {},
            },
        },
        "required": ["agent_type", "request"],
    }

    async def execute(
        self, agent_type: str, request: str, context: dict | None = None
    ) -> ToolResult:
        try:
            from agents.orchestrator.advanced import AgentOrchestrator

            orchestrator = AgentOrchestrator()
            await orchestrator.initialize()

            result = await orchestrator.process_with_agent(
                agent_type=agent_type, request=request, context=context or {}
            )

            if result["success"]:
                return ToolResult(
                    success=True,
                    data={
                        "agent_type": agent_type,
                        "response": result["response"],
                        "confidence": result.get("confidence", 0.0),
                    },
                )
            else:
                return ToolResult(
                    success=False, data=None, error=result.get("error", "Agent execution failed")
                )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ListAgentsTool(BaseTool):
    """List available specialized agents"""

    name = "list_agents"
    description = "List all available specialized agents and their capabilities"
    parameters = {"type": "object", "properties": {}}

    async def execute(self) -> ToolResult:
        agents = {
            "code_review": {
                "description": "Analyzes code for issues, optimizations, and improvements",
                "capabilities": [
                    "code analysis",
                    "security review",
                    "performance optimization",
                    "bug detection",
                ],
            },
            "research": {
                "description": "Conducts research and synthesizes information from multiple sources",
                "capabilities": [
                    "web search",
                    "information synthesis",
                    "fact checking",
                    "summarization",
                ],
            },
            "creative": {
                "description": "Generates creative content like stories, poems, and ideas",
                "capabilities": ["story writing", "poetry", "brainstorming", "content generation"],
            },
            "planning": {
                "description": "Creates structured plans and breaks down complex tasks",
                "capabilities": [
                    "project planning",
                    "task breakdown",
                    "scheduling",
                    "goal setting",
                ],
            },
        }
        return ToolResult(success=True, data={"agents": agents})


class CreateWorkflowTool(BaseTool):
    """Create an automated workflow"""

    name = "create_workflow"
    description = "Create a new automated workflow with triggers and actions"
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name for the workflow"},
            "trigger_type": {
                "type": "string",
                "enum": ["time", "event", "condition"],
                "description": "Type of trigger for the workflow",
            },
            "trigger_config": {
                "type": "object",
                "description": (
                    "Configuration for the trigger "
                    "(e.g., {'interval_minutes': 60} for time trigger)"
                ),
            },
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of actions to execute when triggered",
            },
        },
        "required": ["name", "trigger_type", "trigger_config", "actions"],
    }

    async def execute(
        self, name: str, trigger_type: str, trigger_config: dict, actions: list[str]
    ) -> ToolResult:
        try:
            from core.automation.triggers import TimeTrigger, TriggerManager

            trigger_manager = TriggerManager()

            trigger_id = f"workflow_{name}_{hash(name) % 10000}"

            if trigger_type == "time":
                from datetime import timedelta

                # Get interval or use default
                interval_minutes = trigger_config.get("interval_minutes", 60)
                interval = timedelta(minutes=interval_minutes)

                trigger = TimeTrigger(
                    trigger_id=trigger_id,
                    name=name,
                    workflow_id=trigger_id,
                    interval=interval,
                    cron_expression=trigger_config.get("cron_expression"),
                )
                await trigger_manager.register_trigger(trigger)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Trigger type '{trigger_type}' not yet implemented",
                )

            return ToolResult(
                success=True,
                data={
                    "workflow_id": trigger_id,
                    "name": name,
                    "trigger_type": trigger_type,
                    "actions": actions,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ListWorkflowsTool(BaseTool):
    """List all active workflows"""

    name = "list_workflows"
    description = "List all active automated workflows"
    parameters = {"type": "object", "properties": {}}

    async def execute(self) -> ToolResult:
        try:
            from core.automation.triggers import TriggerManager

            trigger_manager = TriggerManager()
            triggers = trigger_manager.list_triggers()

            workflows = []
            for trigger_id, trigger in triggers.items():
                if trigger_id.startswith("workflow_"):
                    workflows.append(
                        {
                            "id": trigger_id,
                            "enabled": trigger.enabled,
                            "trigger_count": trigger.trigger_count,
                            "last_triggered": trigger.last_triggered,
                        }
                    )

            return ToolResult(success=True, data={"workflows": workflows})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class CreateUserTool(BaseTool):
    """Create a new user profile"""

    name = "create_user"
    description = "Create a new user profile for multi-user support"
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Display name for the user"},
            "permission_level": {
                "type": "string",
                "enum": ["guest", "user", "power_user", "admin"],
                "description": "Permission level for the user",
                "default": "user",
            },
            "preferences": {
                "type": "object",
                "description": "User preferences (e.g., {'language': 'en', 'formality': 'casual'})",
            },
        },
        "required": ["name"],
    }

    async def execute(
        self, name: str, permission_level: str = "user", preferences: dict | None = None
    ) -> ToolResult:
        try:
            from pathlib import Path

            from core.multi_user.user_manager import UserManager

            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            user_manager = UserManager(data_dir / "users.db")

            user_id = await user_manager.create_user(
                name=name, permission_level=permission_level, preferences=preferences or {}
            )

            return ToolResult(
                success=True,
                data={"user_id": user_id, "name": name, "permission_level": permission_level},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetUsersTool(BaseTool):
    """List all users"""

    name = "get_users"
    description = "List all registered users"
    parameters = {"type": "object", "properties": {}}

    async def execute(self) -> ToolResult:
        try:
            from pathlib import Path

            from core.multi_user.user_manager import UserManager

            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            user_manager = UserManager(data_dir / "users.db")

            users = await user_manager.list_users()

            return ToolResult(success=True, data={"users": [user.to_dict() for user in users]})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetUserStatsTool(BaseTool):
    """Get user statistics and activity"""

    name = "get_user_stats"
    description = "Get statistics and activity for a specific user"
    parameters = {
        "type": "object",
        "properties": {"user_id": {"type": "string", "description": "ID of the user"}},
        "required": ["user_id"],
    }

    async def execute(self, user_id: str) -> ToolResult:
        try:
            from pathlib import Path

            from core.multi_user.user_manager import UserManager

            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            user_manager = UserManager(data_dir / "users.db")

            stats = await user_manager.get_user_stats(user_id)

            return ToolResult(success=True, data=stats)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class RecallEpisodicMemoryTool(BaseTool):
    """Recall past conversations and experiences"""

    name = "recall_episodic_memory"
    description = "Recall past conversations and experiences from episodic memory"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for memories"},
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return",
                "default": 5,
            },
            "memory_type": {
                "type": "string",
                "enum": ["conversation", "action", "observation", "learning", "all"],
                "description": "Type of memory to search for",
                "default": "all",
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, limit: int = 5, memory_type: str = "all") -> ToolResult:
        try:
            from core.memory.episodic import EpisodeQuery, EpisodeType, EpisodicMemory

            episodic = EpisodicMemory()
            await episodic.initialize()

            # Build query based on memory type
            episode_types = None
            keywords = [query]

            if memory_type == "conversation":
                episode_types = [EpisodeType.CONVERSATION]
            elif memory_type == "action":
                episode_types = [EpisodeType.ACTION]
            elif memory_type == "learning":
                episode_types = [EpisodeType.LEARNING]
            elif memory_type == "observation":
                episode_types = [EpisodeType.OBSERVATION]
            elif memory_type == "decision":
                episode_types = [EpisodeType.DECISION]
            elif memory_type == "error":
                episode_types = [EpisodeType.ERROR]

            # Create query and retrieve episodes
            query_obj = EpisodeQuery(
                episode_types=episode_types,
                keywords=keywords,
                limit=limit,
            )
            episodes = await episodic.retrieve_episodes(query_obj)

            # Convert episodes to dict format
            memories = [ep.to_dict() for ep in episodes]

            return ToolResult(success=True, data={"memories": memories})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetSuggestionsTool(BaseTool):
    """Get smart suggestions based on user patterns"""

    name = "get_suggestions"
    description = "Get smart, context-aware suggestions based on user behavior patterns"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "ID of the user to get suggestions for"},
            "context": {
                "type": "object",
                "description": "Current context (time_of_day, active_applications, etc.)",
            },
        },
        "required": ["user_id"],
    }

    async def execute(self, user_id: str, context: dict | None = None) -> ToolResult:
        try:
            from core.prediction.suggestion_engine import SmartSuggestionEngine

            suggestion_engine = SmartSuggestionEngine()

            suggestions = await suggestion_engine.generate_suggestions(
                user_id=user_id, context=context or {}
            )

            return ToolResult(success=True, data={"suggestions": suggestions})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class CheckAnomaliesTool(BaseTool):
    """Check for system or behavior anomalies"""

    name = "check_anomalies"
    description = "Check for unusual system behavior or security anomalies"
    parameters = {
        "type": "object",
        "properties": {
            "check_type": {
                "type": "string",
                "enum": ["system", "behavior", "all"],
                "description": "Type of anomaly check to perform",
                "default": "all",
            }
        },
    }

    async def execute(self, check_type: str = "all") -> ToolResult:
        try:
            from core.prediction.anomaly_detector import AnomalyDetector

            detector = AnomalyDetector()

            results = {}

            if check_type in ["system", "all"]:
                # Check system health
                results["system"] = await detector.check_system_health({})

            if check_type in ["behavior", "all"]:
                # Check user behavior (would need user_id in real implementation)
                results["behavior"] = {"status": "not_implemented"}

            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetConversationHistoryTool(BaseTool):
    """Get recent conversation history"""

    name = "get_conversation_history"
    description = "Get recent conversation history"
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of conversations to return",
                "default": 10,
            },
            "user_id": {"type": "string", "description": "Filter by specific user ID"},
        },
    }

    async def execute(self, limit: int = 10, user_id: str | None = None) -> ToolResult:
        try:
            from core.memory.episodic import EpisodicMemory

            episodic = EpisodicMemory()
            await episodic.initialize()

            # Use recall_conversations method which is the proper way to get conversations
            episodes = await episodic.recall_conversations(
                about=None, with_participant=user_id, limit=limit
            )

            # Convert episodes to dict format
            conversations = [ep.to_dict() for ep in episodes]

            return ToolResult(success=True, data={"conversations": conversations})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
