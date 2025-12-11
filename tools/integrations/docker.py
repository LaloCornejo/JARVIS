from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from tools.base import BaseTool, ToolResult


class DockerClient:
    def __init__(self, docker_host: str | None = None):
        self.docker_host = docker_host or os.environ.get("DOCKER_HOST", "")
        self._docker_path = "docker"

    async def _run_command(self, *args: str) -> tuple[bool, str]:
        cmd = [self._docker_path, *args]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return True, stdout.decode().strip()
            return False, stderr.decode().strip()
        except Exception as e:
            return False, str(e)

    async def list_containers(self, all_containers: bool = False) -> list[dict[str, Any]]:
        args = ["ps", "--format", "{{json .}}"]
        if all_containers:
            args.insert(1, "-a")
        success, output = await self._run_command(*args)
        if not success:
            return []
        containers = []
        for line in output.strip().split("\n"):
            if line:
                containers.append(json.loads(line))
        return containers

    async def list_images(self) -> list[dict[str, Any]]:
        success, output = await self._run_command("images", "--format", "{{json .}}")
        if not success:
            return []
        images = []
        for line in output.strip().split("\n"):
            if line:
                images.append(json.loads(line))
        return images

    async def start_container(self, container_id: str) -> tuple[bool, str]:
        return await self._run_command("start", container_id)

    async def stop_container(self, container_id: str) -> tuple[bool, str]:
        return await self._run_command("stop", container_id)

    async def restart_container(self, container_id: str) -> tuple[bool, str]:
        return await self._run_command("restart", container_id)

    async def remove_container(self, container_id: str, force: bool = False) -> tuple[bool, str]:
        args = ["rm"]
        if force:
            args.append("-f")
        args.append(container_id)
        return await self._run_command(*args)

    async def logs(self, container_id: str, tail: int = 100) -> tuple[bool, str]:
        return await self._run_command("logs", "--tail", str(tail), container_id)

    async def exec_command(self, container_id: str, command: str) -> tuple[bool, str]:
        return await self._run_command("exec", container_id, "sh", "-c", command)

    async def inspect(self, container_id: str) -> dict[str, Any] | None:
        success, output = await self._run_command("inspect", container_id)
        if success:
            data = json.loads(output)
            return data[0] if data else None
        return None

    async def run(
        self,
        image: str,
        name: str | None = None,
        ports: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        detach: bool = True,
    ) -> tuple[bool, str]:
        args = ["run"]
        if detach:
            args.append("-d")
        if name:
            args.extend(["--name", name])
        if ports:
            for host_port, container_port in ports.items():
                args.extend(["-p", f"{host_port}:{container_port}"])
        if volumes:
            for host_path, container_path in volumes.items():
                args.extend(["-v", f"{host_path}:{container_path}"])
        if env:
            for key, value in env.items():
                args.extend(["-e", f"{key}={value}"])
        args.append(image)
        return await self._run_command(*args)

    async def pull(self, image: str) -> tuple[bool, str]:
        return await self._run_command("pull", image)

    async def build(self, path: str, tag: str | None = None) -> tuple[bool, str]:
        args = ["build"]
        if tag:
            args.extend(["-t", tag])
        args.append(path)
        return await self._run_command(*args)

    async def stats(self, container_id: str | None = None) -> list[dict[str, Any]]:
        args = ["stats", "--no-stream", "--format", "{{json .}}"]
        if container_id:
            args.append(container_id)
        success, output = await self._run_command(*args)
        if not success:
            return []
        stats = []
        for line in output.strip().split("\n"):
            if line:
                stats.append(json.loads(line))
        return stats


_docker_client: DockerClient | None = None


def get_docker_client() -> DockerClient:
    global _docker_client
    if _docker_client is None:
        _docker_client = DockerClient()
    return _docker_client


class DockerListContainersTool(BaseTool):
    name = "docker_list_containers"
    description = "List Docker containers. Optionally include stopped containers."
    parameters = {
        "type": "object",
        "properties": {
            "all": {
                "type": "boolean",
                "description": "Include stopped containers (default: false)",
            },
        },
        "required": [],
    }

    async def execute(self, all: bool = False) -> ToolResult:
        client = get_docker_client()
        try:
            containers = await client.list_containers(all_containers=all)
            return ToolResult(success=True, data=containers)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerListImagesTool(BaseTool):
    name = "docker_list_images"
    description = "List Docker images"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_docker_client()
        try:
            images = await client.list_images()
            return ToolResult(success=True, data=images)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerStartTool(BaseTool):
    name = "docker_start"
    description = "Start a stopped Docker container"
    parameters = {
        "type": "object",
        "properties": {
            "container": {
                "type": "string",
                "description": "Container ID or name",
            },
        },
        "required": ["container"],
    }

    async def execute(self, container: str) -> ToolResult:
        client = get_docker_client()
        try:
            success, msg = await client.start_container(container)
            if success:
                return ToolResult(success=True, data=f"Started container: {container}")
            return ToolResult(success=False, data=None, error=msg)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerStopTool(BaseTool):
    name = "docker_stop"
    description = "Stop a running Docker container"
    parameters = {
        "type": "object",
        "properties": {
            "container": {
                "type": "string",
                "description": "Container ID or name",
            },
        },
        "required": ["container"],
    }

    async def execute(self, container: str) -> ToolResult:
        client = get_docker_client()
        try:
            success, msg = await client.stop_container(container)
            if success:
                return ToolResult(success=True, data=f"Stopped container: {container}")
            return ToolResult(success=False, data=None, error=msg)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerRestartTool(BaseTool):
    name = "docker_restart"
    description = "Restart a Docker container"
    parameters = {
        "type": "object",
        "properties": {
            "container": {
                "type": "string",
                "description": "Container ID or name",
            },
        },
        "required": ["container"],
    }

    async def execute(self, container: str) -> ToolResult:
        client = get_docker_client()
        try:
            success, msg = await client.restart_container(container)
            if success:
                return ToolResult(success=True, data=f"Restarted container: {container}")
            return ToolResult(success=False, data=None, error=msg)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerLogsTool(BaseTool):
    name = "docker_logs"
    description = "Get logs from a Docker container"
    parameters = {
        "type": "object",
        "properties": {
            "container": {
                "type": "string",
                "description": "Container ID or name",
            },
            "tail": {
                "type": "integer",
                "description": "Number of lines to show from end (default: 100)",
            },
        },
        "required": ["container"],
    }

    async def execute(self, container: str, tail: int = 100) -> ToolResult:
        client = get_docker_client()
        try:
            success, output = await client.logs(container, tail)
            if success:
                return ToolResult(success=True, data=output)
            return ToolResult(success=False, data=None, error=output)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerExecTool(BaseTool):
    name = "docker_exec"
    description = "Execute a command inside a running Docker container"
    parameters = {
        "type": "object",
        "properties": {
            "container": {
                "type": "string",
                "description": "Container ID or name",
            },
            "command": {
                "type": "string",
                "description": "Command to execute",
            },
        },
        "required": ["container", "command"],
    }

    async def execute(self, container: str, command: str) -> ToolResult:
        client = get_docker_client()
        try:
            success, output = await client.exec_command(container, command)
            if success:
                return ToolResult(success=True, data=output)
            return ToolResult(success=False, data=None, error=output)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerRunTool(BaseTool):
    name = "docker_run"
    description = "Run a new Docker container from an image"
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Docker image to run",
            },
            "name": {
                "type": "string",
                "description": "Name for the container",
            },
            "ports": {
                "type": "object",
                "description": "Port mappings as {host_port: container_port}",
            },
            "volumes": {
                "type": "object",
                "description": "Volume mappings as {host_path: container_path}",
            },
            "env": {
                "type": "object",
                "description": "Environment variables as {key: value}",
            },
        },
        "required": ["image"],
    }

    async def execute(
        self,
        image: str,
        name: str | None = None,
        ports: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ToolResult:
        client = get_docker_client()
        try:
            success, output = await client.run(
                image, name=name, ports=ports, volumes=volumes, env=env
            )
            if success:
                return ToolResult(
                    success=True,
                    data={"container_id": output, "image": image, "name": name},
                )
            return ToolResult(success=False, data=None, error=output)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerPullTool(BaseTool):
    name = "docker_pull"
    description = "Pull a Docker image from a registry"
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image name (e.g., 'nginx:latest')",
            },
        },
        "required": ["image"],
    }

    async def execute(self, image: str) -> ToolResult:
        client = get_docker_client()
        try:
            success, output = await client.pull(image)
            if success:
                return ToolResult(success=True, data=f"Pulled image: {image}")
            return ToolResult(success=False, data=None, error=output)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerStatsTool(BaseTool):
    name = "docker_stats"
    description = "Get resource usage statistics for Docker containers"
    parameters = {
        "type": "object",
        "properties": {
            "container": {
                "type": "string",
                "description": "Container ID or name (optional, shows all if omitted)",
            },
        },
        "required": [],
    }

    async def execute(self, container: str | None = None) -> ToolResult:
        client = get_docker_client()
        try:
            stats = await client.stats(container)
            return ToolResult(success=True, data=stats)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DockerInspectTool(BaseTool):
    name = "docker_inspect"
    description = "Get detailed information about a Docker container"
    parameters = {
        "type": "object",
        "properties": {
            "container": {
                "type": "string",
                "description": "Container ID or name",
            },
        },
        "required": ["container"],
    }

    async def execute(self, container: str) -> ToolResult:
        client = get_docker_client()
        try:
            info = await client.inspect(container)
            if info:
                return ToolResult(success=True, data=info)
            return ToolResult(success=False, data=None, error=f"Container not found: {container}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
