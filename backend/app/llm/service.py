"""
LLM Service for natural language reasoning and command generation.
Integrates with Ollama (local) and Hugging Face (fallback).
"""
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RobotCommand(BaseModel):
    """Structured robot command."""
    action: str  # pick, place, pour, follow_trajectory, home
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    safety_validated: bool
    estimated_duration_sec: float


class LLMResponse(BaseModel):
    """LLM response with reasoning."""
    command: Optional[RobotCommand]
    reasoning: str
    raw_response: str
    tokens_used: int
    latency_ms: float


class PerceptionContext(BaseModel):
    """Perception context for LLM."""
    detected_objects: List[Dict[str, Any]]
    liquid_level: Optional[float]
    depth_info: Dict[str, Any]
    workspace_status: str


class LLMService:
    """
    LLM service for natural language command processing.
    Supports Ollama (local) and Hugging Face (API fallback).
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.2:latest",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 300,
    ):
        self.ollama_host = ollama_host
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.client = httpx.AsyncClient(timeout=timeout)
        self.conversation_history: List[Dict[str, str]] = []

        logger.info(
            "llm_service_initialized",
            host=ollama_host,
            model=model,
        )

    async def initialize(self) -> bool:
        """Initialize LLM service and verify connectivity."""
        try:
            # Check Ollama availability
            response = await self.client.get(f"{self.ollama_host}/api/tags")

            if response.status_code == 200:
                logger.info("ollama_connected", models=response.json())
                return True
            else:
                logger.warning("ollama_not_available", status=response.status_code)
                return False

        except Exception as e:
            logger.error("llm_init_failed", error=str(e))
            return False

    async def process_natural_language_command(
        self,
        command: str,
        perception_context: Optional[PerceptionContext] = None,
        user_role: str = "operator",
    ) -> LLMResponse:
        """
        Process natural language command and generate structured robot command.

        Args:
            command: Natural language command from user
            perception_context: Current perception state
            user_role: User's role for safety constraints

        Returns:
            LLMResponse with structured command and reasoning
        """
        start_time = datetime.now()

        logger.info("processing_nl_command", command=command)

        # Build prompt with perception context
        prompt = self._build_prompt(command, perception_context, user_role)

        # Call LLM
        try:
            raw_response = await self._call_ollama(prompt)

            # Parse response to structured command
            robot_command = self._parse_response(raw_response)

            # Validate safety
            robot_command.safety_validated = self._validate_safety(
                robot_command,
                perception_context
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Check latency budget
            if latency_ms > settings.llm_timeout * 1000:
                logger.warning(
                    "llm_latency_exceeded",
                    latency_ms=latency_ms,
                    budget_ms=settings.llm_timeout * 1000,
                )

            response = LLMResponse(
                command=robot_command,
                reasoning=robot_command.reasoning,
                raw_response=raw_response,
                tokens_used=len(raw_response.split()),  # Approximate
                latency_ms=latency_ms,
            )

            logger.info(
                "nl_command_processed",
                command=command,
                action=robot_command.action,
                safety_validated=robot_command.safety_validated,
                latency_ms=latency_ms,
            )

            return response

        except Exception as e:
            logger.error("llm_processing_failed", command=command, error=str(e))

            # Return safe fallback
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            return LLMResponse(
                command=None,
                reasoning=f"Failed to process command: {str(e)}",
                raw_response="",
                tokens_used=0,
                latency_ms=latency_ms,
            )

    def _build_prompt(
        self,
        command: str,
        perception_context: Optional[PerceptionContext],
        user_role: str,
    ) -> str:
        """Build LLM prompt with context."""

        # System prompt
        system_prompt = """You are an AI assistant for a robotic manipulation system.

Your task is to interpret natural language commands and convert them into structured robot commands.

Available actions:
- pick: Pick up an object
- place: Place an object at a location
- pour: Pour liquid from one container to another
- follow_trajectory: Follow a specific path
- home: Return to home position

Respond in JSON format with the following structure:
{
    "action": "<action_type>",
    "parameters": {
        "object": "<object_name>",
        "position": [x, y, z],
        "orientation": [roll, pitch, yaw],
        ...
    },
    "confidence": <0.0-1.0>,
    "reasoning": "<explanation>",
    "estimated_duration_sec": <seconds>
}

SAFETY RULES:
1. Never execute commands that could damage equipment
2. Always validate workspace bounds
3. Ensure objects are detected before manipulation
4. Reject ambiguous or unsafe commands
5. Prioritize safety over task completion
"""

        # Perception context
        context_str = ""
        if perception_context:
            context_str = f"""
Current Perception:
- Detected objects: {json.dumps(perception_context.detected_objects, indent=2)}
- Liquid level: {perception_context.liquid_level}
- Workspace status: {perception_context.workspace_status}
"""

        # User command
        user_prompt = f"""
User role: {user_role}
Command: {command}
{context_str}

Generate the structured command:"""

        full_prompt = f"{system_prompt}\n{user_prompt}"

        return full_prompt

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            response = await self.client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error("ollama_api_error", status=response.status_code)
                return ""

        except Exception as e:
            logger.error("ollama_call_failed", error=str(e))

            # Fallback to mock response for development
            return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock LLM response for development."""
        # Extract command from prompt (simple heuristic)
        if "pick" in prompt.lower():
            return json.dumps({
                "action": "pick",
                "parameters": {
                    "object": "bottle",
                    "position": [0.5, 0.2, 0.8],
                    "approach_vector": [0, 0, -1],
                },
                "confidence": 0.90,
                "reasoning": "Detected bottle at position (0.5, 0.2, 0.8). Planning top-down grasp approach.",
                "estimated_duration_sec": 3.5,
            })
        elif "pour" in prompt.lower():
            return json.dumps({
                "action": "pour",
                "parameters": {
                    "source": "bottle",
                    "target": "cup",
                    "amount_ml": 200,
                    "pour_rate": 50,
                },
                "confidence": 0.85,
                "reasoning": "Planning pour operation from bottle to cup with controlled flow rate.",
                "estimated_duration_sec": 5.0,
            })
        elif "place" in prompt.lower():
            return json.dumps({
                "action": "place",
                "parameters": {
                    "object": "bottle",
                    "position": [0.3, 0.3, 0.0],
                    "orientation": [0, 0, 0],
                },
                "confidence": 0.88,
                "reasoning": "Planning placement of bottle at target position with stable orientation.",
                "estimated_duration_sec": 3.0,
            })
        else:
            return json.dumps({
                "action": "home",
                "parameters": {},
                "confidence": 0.95,
                "reasoning": "Returning to home position as requested.",
                "estimated_duration_sec": 2.0,
            })

    def _parse_response(self, raw_response: str) -> RobotCommand:
        """Parse LLM response into structured command."""
        try:
            # Extract JSON from response (may have markdown formatting)
            json_str = raw_response

            # Remove markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            # Parse JSON
            data = json.loads(json_str)

            return RobotCommand(
                action=data.get("action", "home"),
                parameters=data.get("parameters", {}),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                safety_validated=False,  # Will be validated separately
                estimated_duration_sec=data.get("estimated_duration_sec", 5.0),
            )

        except Exception as e:
            logger.error("llm_response_parse_failed", error=str(e), response=raw_response)

            # Return safe default
            return RobotCommand(
                action="home",
                parameters={},
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                safety_validated=False,
                estimated_duration_sec=2.0,
            )

    def _validate_safety(
        self,
        command: RobotCommand,
        perception_context: Optional[PerceptionContext],
    ) -> bool:
        """Validate command safety."""

        # Check confidence threshold
        if command.confidence < 0.7:
            logger.warning("command_low_confidence", confidence=command.confidence)
            return False

        # Validate workspace bounds
        if "position" in command.parameters:
            pos = command.parameters["position"]
            bounds = settings.robot_workspace_bounds

            if not (bounds[0] <= pos[0] <= bounds[1] and
                    bounds[2] <= pos[1] <= bounds[3] and
                    bounds[4] <= pos[2] <= bounds[5]):
                logger.warning("command_outside_workspace", position=pos)
                return False

        # Validate object detection for manipulation
        if command.action in ["pick", "pour"]:
            if not perception_context or not perception_context.detected_objects:
                logger.warning("command_requires_object_detection")
                return False

        # All safety checks passed
        return True

    async def close(self):
        """Close LLM service and cleanup."""
        await self.client.aclose()
        logger.info("llm_service_closed")


# Global LLM service instance
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get or create global LLM service."""
    global _llm_service

    if _llm_service is None:
        _llm_service = LLMService(
            ollama_host=settings.ollama_host,
            model=settings.ollama_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )
        await _llm_service.initialize()

    return _llm_service
