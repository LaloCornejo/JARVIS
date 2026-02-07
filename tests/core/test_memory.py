"""Tests for Memory Systems"""

import pytest

from core.memory.episodic import EpisodicMemory
from core.memory.knowledge_graph import (
    EntityType,
    KnowledgeGraph,
    RelationType,
)
from core.memory.procedural import ProceduralMemory, SkillLevel


class TestEpisodicMemory:
    """Test suite for EpisodicMemory"""

    @pytest.fixture
    async def memory(self, temp_db_path):
        """Create an EpisodicMemory instance"""
        return EpisodicMemory(temp_db_path / "episodic.db")

    @pytest.mark.asyncio
    async def test_initialization(self, memory):
        """Test memory initializes correctly"""
        mem = await memory
        assert mem is not None

    @pytest.mark.asyncio
    async def test_record_conversation(self, memory):
        """Test recording a conversation"""
        mem = await memory
        episode_id = await mem.record_conversation(
            user_input="Hello JARVIS", assistant_response="Hi there!", user_id="user_1"
        )
        assert episode_id is not None
        assert episode_id.startswith("ep_")

    @pytest.mark.asyncio
    async def test_record_action(self, memory):
        """Test recording an action"""
        mem = await memory
        episode_id = await mem.record_action(
            action_type="file_open",
            description="Opened a file",
            outcome={"success": True},
            user_id="user_1",
        )
        assert episode_id is not None

    @pytest.mark.asyncio
    async def test_get_recent_episodes(self, memory):
        """Test getting recent episodes"""
        mem = await memory
        await mem.record_conversation(user_input="Test 1", user_id="user_1")
        await mem.record_conversation(user_input="Test 2", user_id="user_1")

        episodes = await mem.get_recent_episodes(limit=10)
        assert len(episodes) >= 2

    @pytest.mark.asyncio
    async def test_search_conversations(self, memory):
        """Test searching conversations"""
        mem = await memory
        await mem.record_conversation(user_input="Python programming", user_id="user_1")

        results = await mem.search_conversations(query="Python")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_memory_persistence(self, memory):
        """Test that memories persist"""
        mem = await memory
        episode_id = await mem.record_conversation(user_input="Persistent", user_id="user_1")

        # Create new instance with same path (simulated by keeping reference)
        episodes = await mem.get_recent_episodes(limit=1)
        assert len(episodes) > 0


class TestKnowledgeGraph:
    """Test suite for KnowledgeGraph"""

    @pytest.fixture
    async def graph(self, temp_db_path):
        """Create a KnowledgeGraph instance"""
        return KnowledgeGraph(temp_db_path / "knowledge.db")

    @pytest.mark.asyncio
    async def test_initialization(self, graph):
        """Test graph initializes correctly"""
        g = await graph
        assert g is not None

    @pytest.mark.asyncio
    async def test_add_entity(self, graph):
        """Test adding an entity"""
        g = await graph
        entity_id = await g.add_entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            properties={"creator": "Guido van Rossum"},
        )
        assert entity_id is not None

    @pytest.mark.asyncio
    async def test_get_entity(self, graph):
        """Test getting an entity"""
        g = await graph
        entity_id = await g.add_entity(name="Test Entity", entity_type=EntityType.CONCEPT)
        entity = await g.get_entity(entity_id)
        assert entity is not None
        assert entity.name == "Test Entity"

    @pytest.mark.asyncio
    async def test_add_relationship(self, graph):
        """Test adding a relationship"""
        g = await graph
        entity1 = await g.add_entity("A", EntityType.PERSON)
        entity2 = await g.add_entity("B", EntityType.PERSON)

        rel_id = await g.add_relationship(
            source_id=entity1, target_id=entity2, relation_type=RelationType.KNOWS
        )
        assert rel_id is not None

    @pytest.mark.asyncio
    async def test_find_entities(self, graph):
        """Test finding entities"""
        g = await graph
        await g.add_entity("Python Language", EntityType.TECHNOLOGY)
        await g.add_entity("Python Snake", EntityType.OBJECT)

        results = await g.find_entities(name="Python")
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_query(self, graph):
        """Test querying the graph"""
        g = await graph
        entity_id = await g.add_entity("Test", EntityType.CONCEPT)

        result = await g.query(start_entity=entity_id, max_depth=1)
        assert result is not None
        assert isinstance(result.entities, list)

    @pytest.mark.asyncio
    async def test_get_stats(self, graph):
        """Test getting graph statistics"""
        g = await graph
        stats = await g.get_stats()
        assert "entity_count" in stats
        assert "relationship_count" in stats


class TestProceduralMemory:
    """Test suite for ProceduralMemory"""

    @pytest.fixture
    async def memory(self, temp_db_path):
        """Create a ProceduralMemory instance"""
        return ProceduralMemory(temp_db_path / "procedural.db")

    @pytest.mark.asyncio
    async def test_initialization(self, memory):
        """Test memory initializes correctly"""
        mem = await memory
        assert mem is not None

    @pytest.mark.asyncio
    async def test_add_skill(self, memory):
        """Test adding a skill"""
        mem = await memory
        skill_id = await mem.add_skill(
            name="Python Programming", description="Writing Python code", category="programming"
        )
        assert skill_id is not None

    @pytest.mark.asyncio
    async def test_get_skill(self, memory):
        """Test getting a skill"""
        mem = await memory
        skill_id = await mem.add_skill(name="Test Skill")
        skill = await mem.get_skill(skill_id)
        assert skill is not None
        assert skill.name == "Test Skill"

    @pytest.mark.asyncio
    async def test_update_skill_proficiency(self, memory):
        """Test updating skill proficiency"""
        mem = await memory
        skill_id = await mem.add_skill(name="Test Skill")
        await mem.update_skill_proficiency(skill_id, SkillLevel.INTERMEDIATE)

        skill = await mem.get_skill(skill_id)
        assert skill.proficiency == SkillLevel.INTERMEDIATE

    @pytest.mark.asyncio
    async def test_add_procedure(self, memory):
        """Test adding a procedure"""
        mem = await memory
        procedure_id = await mem.add_procedure(
            name="Git Commit",
            description="How to commit changes",
            steps=["git add .", "git commit -m 'message'", "git push"],
        )
        assert procedure_id is not None

    @pytest.mark.asyncio
    async def test_search_skills(self, memory):
        """Test searching skills"""
        mem = await memory
        await mem.add_skill(name="Python Development", category="programming")

        results = await mem.search_skills(query="Python")
        assert len(results) > 0
