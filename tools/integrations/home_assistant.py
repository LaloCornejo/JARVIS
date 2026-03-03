from __future__ import annotations

import os
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult


class HomeAssistantClient:
    def __init__(self, ha_url: str | None = None, ha_token: str | None = None):
        self.ha_url = ha_url or os.environ.get("HA_URL") or os.environ.get("HOMEASSISTANT_URL")
        self.ha_token = (
            ha_token or os.environ.get("HA_TOKEN") or os.environ.get("HOMEASSISTANT_TOKEN")
        )
        if not self.ha_url:
            raise ValueError(
                "Home Assistant URL not configured. Set HA_URL or HOMEASSISTANT_URL env var."
            )
        if not self.ha_token:
            raise ValueError(
                "Home Assistant token not configured. Set HA_TOKEN or HOMEASSISTANT_TOKEN env var."
            )
        self.ha_url = self.ha_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.ha_token}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_states(self) -> list[dict[str, Any]]:
        client = await self._get_client()
        response = await client.get(f"{self.ha_url}/api/states")
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to get states: {response.status_code} {response.text}")

    async def get_state(self, entity_id: str) -> dict[str, Any] | None:
        client = await self._get_client()
        response = await client.get(f"{self.ha_url}/api/states/{entity_id}")
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            return None
        raise Exception(f"Failed to get state: {response.status_code} {response.text}")

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: dict[str, Any] | None = None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        client = await self._get_client()
        data = service_data or {}
        if entity_id:
            data["entity_id"] = entity_id
        response = await client.post(
            f"{self.ha_url}/api/services/{domain}/{service}",
            json=data,
        )
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to call service: {response.status_code} {response.text}")

    async def get_services(self) -> dict[str, Any]:
        client = await self._get_client()
        response = await client.get(f"{self.ha_url}/api/services")
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to get services: {response.status_code} {response.text}")

    async def get_config(self) -> dict[str, Any]:
        client = await self._get_client()
        response = await client.get(f"{self.ha_url}/api/config")
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to get config: {response.status_code} {response.text}")

    async def get_devices(self) -> list[dict[str, Any]]:
        client = await self._get_client()
        response = await client.get(f"{self.ha_url}/api/devices")
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to get devices: {response.status_code} {response.text}")

    async def get_entities_by_domain(self, domain: str) -> list[dict[str, Any]]:
        states = await self.get_states()
        return [s for s in states if s["entity_id"].startswith(f"{domain}.")]


_ha_client: HomeAssistantClient | None = None


def get_ha_client() -> HomeAssistantClient:
    global _ha_client
    if _ha_client is None:
        _ha_client = HomeAssistantClient()
    return _ha_client


class HomeAssistantListEntitiesTool(BaseTool):
    name = "homeassistant_list_entities"
    description = "List all entities in Home Assistant. Optionally filter by domain (light, switch, sensor, climate, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "Optional domain filter (e.g., 'light', 'switch', 'sensor', 'climate', 'cover', 'fan')",
            },
        },
        "required": [],
    }

    async def execute(self, domain: str | None = None) -> ToolResult:
        client = get_ha_client()
        try:
            if domain:
                entities = await client.get_entities_by_domain(domain)
            else:
                entities = await client.get_states()
            return ToolResult(success=True, data=entities)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class HomeAssistantGetStateTool(BaseTool):
    name = "homeassistant_get_state"
    description = "Get the current state of a specific Home Assistant entity"
    parameters = {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "The entity ID (e.g., 'light.living_room', 'switch.garage_door', 'climate.thermostat')",
            },
        },
        "required": ["entity_id"],
    }

    async def execute(self, entity_id: str) -> ToolResult:
        client = get_ha_client()
        try:
            state = await client.get_state(entity_id)
            if state:
                return ToolResult(success=True, data=state)
            return ToolResult(success=False, data=None, error=f"Entity not found: {entity_id}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class HomeAssistantCallServiceTool(BaseTool):
    name = "homeassistant_call_service"
    description = "Call a Home Assistant service (e.g., turn on/off lights, set thermostat, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "The domain of the service (e.g., 'light', 'switch', 'climate', 'cover', 'fan', 'script')",
            },
            "service": {
                "type": "string",
                "description": "The service to call (e.g., 'turn_on', 'turn_off', 'set_temperature', 'toggle')",
            },
            "entity_id": {
                "type": "string",
                "description": "The entity ID to target (e.g., 'light.living_room')",
            },
            "service_data": {
                "type": "object",
                "description": "Additional service data as key-value pairs (e.g., {'brightness': 150, 'color_name': 'red'})",
            },
        },
        "required": ["domain", "service", "entity_id"],
    }

    async def execute(
        self,
        domain: str,
        service: str,
        entity_id: str,
        service_data: dict[str, Any] | None = None,
    ) -> ToolResult:
        client = get_ha_client()
        try:
            result = await client.call_service(domain, service, service_data, entity_id)
            return ToolResult(
                success=True,
                data={
                    "domain": domain,
                    "service": service,
                    "entity_id": entity_id,
                    "result": result,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class HomeAssistantTurnOnTool(BaseTool):
    name = "homeassistant_turn_on"
    description = "Turn on a Home Assistant entity (light, switch, fan, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "The entity ID to turn on (e.g., 'light.living_room', 'switch.garage_door')",
            },
            "brightness": {
                "type": "number",
                "description": "Brightness level (0-255) for lights",
            },
            "color_name": {
                "type": "string",
                "description": "Color name for RGB lights (e.g., 'red', 'blue', 'green')",
            },
            "color_temp": {
                "type": "number",
                "description": "Color temperature in Kelvin for tunable lights",
            },
        },
        "required": ["entity_id"],
    }

    async def execute(
        self,
        entity_id: str,
        brightness: int | None = None,
        color_name: str | None = None,
        color_temp: int | None = None,
    ) -> ToolResult:
        client = get_ha_client()
        service_data: dict[str, Any] = {}
        if brightness is not None:
            service_data["brightness"] = brightness
        if color_name:
            service_data["color_name"] = color_name
        if color_temp:
            service_data["color_temp"] = color_temp

        domain = entity_id.split(".")[0]
        try:
            await client.call_service(domain, "turn_on", service_data, entity_id)
            return ToolResult(success=True, data=f"Turned on {entity_id}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class HomeAssistantTurnOffTool(BaseTool):
    name = "homeassistant_turn_off"
    description = "Turn off a Home Assistant entity (light, switch, fan, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "The entity ID to turn off (e.g., 'light.living_room', 'switch.garage_door')",
            },
        },
        "required": ["entity_id"],
    }

    async def execute(self, entity_id: str) -> ToolResult:
        client = get_ha_client()
        domain = entity_id.split(".")[0]
        try:
            await client.call_service(domain, "turn_off", None, entity_id)
            return ToolResult(success=True, data=f"Turned off {entity_id}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class HomeAssistantToggleTool(BaseTool):
    name = "homeassistant_toggle"
    description = "Toggle a Home Assistant entity (light, switch, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "The entity ID to toggle (e.g., 'light.living_room', 'switch.garage_door')",
            },
        },
        "required": ["entity_id"],
    }

    async def execute(self, entity_id: str) -> ToolResult:
        client = get_ha_client()
        domain = entity_id.split(".")[0]
        try:
            await client.call_service(domain, "toggle", None, entity_id)
            return ToolResult(success=True, data=f"Toggled {entity_id}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class HomeAssistantGetServicesTool(BaseTool):
    name = "homeassistant_get_services"
    description = "Get all available services in Home Assistant"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        client = get_ha_client()
        try:
            services = await client.get_services()
            return ToolResult(success=True, data=services)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class HomeAssistantGetConfigTool(BaseTool):
    name = "homeassistant_get_config"
    description = "Get Home Assistant configuration and system info"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        client = get_ha_client()
        try:
            config = await client.get_config()
            return ToolResult(success=True, data=config)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class HomeAssistantGetDevicesTool(BaseTool):
    name = "homeassistant_get_devices"
    description = "Get all devices registered in Home Assistant"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        client = get_ha_client()
        try:
            devices = await client.get_devices()
            return ToolResult(success=True, data=devices)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
