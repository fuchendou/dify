from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from sqlalchemy.orm import sessionmaker

from core.llm_generator.output_parser.structured_output import invoke_llm_with_structured_output
from core.memory.base import BufferMemory
from core.model_manager import ModelInstance
from core.model_runtime.entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageRole,
    SystemPromptMessage,
    UserPromptMessage,
)
from core.workflow.enums import WorkflowNodeExecutionMetadataKey
from extensions.ext_database import db
from extensions.ext_storage import storage
from repositories.api_workflow_node_execution_repository import DifyAPIWorkflowNodeExecutionRepository
from repositories.factory import DifyAPIRepositoryFactory


class NodeLevelMemory(BufferMemory):
    def __init__(
        self,
        *,
        tenant_id: str,
        app_id: str,
        workflow_id: str,
        workflow_run_id: str,
        node_id: str,
        model_instance: ModelInstance,
    ):
        super().__init__(model_instance=model_instance)
        self.tenant_id = tenant_id
        self.app_id = app_id
        self.workflow_id = workflow_id
        self.workflow_run_id = workflow_run_id
        self.node_id = node_id
        self.metadata: dict[WorkflowNodeExecutionMetadataKey, Any] = {}
        self._repository: DifyAPIWorkflowNodeExecutionRepository | None = None
        self._history: list[PromptMessage] | None = None

    @property
    def repository(self) -> DifyAPIWorkflowNodeExecutionRepository:
        if self._repository is None:
            session_maker = sessionmaker(bind=db.engine, expire_on_commit=False)
            self._repository = DifyAPIRepositoryFactory.create_api_workflow_node_execution_repository(session_maker)
        return self._repository

    def _safe_segment(self, raw: str) -> str:
        text = raw or "unknown"
        sanitized = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)
        return sanitized[:128] or "unknown"

    def _context_prefix(self) -> str:
        tenant = self._safe_segment(self.tenant_id)
        app = self._safe_segment(self.app_id)
        workflow = self._safe_segment(self.workflow_id)
        node = self._safe_segment(self.node_id)
        return f"memory/{tenant}/{app}/workflow/{workflow}/node/{node}/"

    def _build_context_uri(self, sha256: str) -> str:
        return f"{self._context_prefix()}{sha256}.json"

    def _load_context_from_ref(self, ref: Mapping[str, Any]) -> Any | None:
        if not ref:
            return None
        uri = ref.get("uri")
        if not uri:
            return None
        prefix = self._context_prefix()
        if not uri.startswith(prefix):
            return None
        if not storage.exists(uri):
            return None
        data = cast(Any, storage.load(uri))
        if isinstance(data, bytes):
            raw_text = data.decode("utf-8")
        elif isinstance(data, str):
            raw_text = data
        elif isinstance(data, Sequence):
            raw_text = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk) for chunk in data)
        else:
            raw_text = str(data)
        try:
            return json.loads(raw_text)
        except Exception:
            return None

    def _deserialize_context(self, context: Any) -> list[PromptMessage]:
        if not context:
            return []

        messages: list[PromptMessage] = []
        for item in context:
            if isinstance(item, PromptMessage):
                messages.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            if role == PromptMessageRole.USER:
                messages.append(UserPromptMessage.model_validate(item))
            elif role == PromptMessageRole.ASSISTANT:
                messages.append(AssistantPromptMessage.model_validate(item))
            elif role == PromptMessageRole.SYSTEM:
                messages.append(SystemPromptMessage.model_validate(item))
        return messages

    def _load_history(self) -> list[PromptMessage]:
        if self._history is not None:
            return self._history

        history: list[PromptMessage] = []
        executions = self.repository.get_executions_by_workflow_run(
            tenant_id=self.tenant_id, app_id=self.app_id, workflow_run_id=self.workflow_run_id
        )

        for execution in executions:
            if execution.node_id != self.node_id:
                continue
            context = None
            metadata = execution.execution_metadata_dict or {}
            context_ref = metadata.get(WorkflowNodeExecutionMetadataKey.CONTEXT_REF.value) or metadata.get(
                WorkflowNodeExecutionMetadataKey.CONTEXT_REF
            )
            if context_ref:
                context = self._load_context_from_ref(context_ref)
            if context is None:
                outputs_dict = execution.outputs_dict or {}
                context = outputs_dict.get("context")
            history.extend(self._deserialize_context(context))

        self._history = history
        return self._history

    def _persist_context(self, messages: Sequence[PromptMessage]) -> Mapping[str, Any] | None:
        if not messages:
            return None
        payload = json.dumps([message.model_dump() for message in messages], ensure_ascii=False, default=str).encode(
            "utf-8"
        )
        sha256 = hashlib.sha256(payload).hexdigest()
        uri = self._build_context_uri(sha256)
        # Idempotent write: only save when the content hash is new for this scope.
        if not storage.exists(uri):
            storage.save(uri, payload)
        return {"uri": uri, "sha256": sha256, "size": len(payload)}

    def append_messages(
        self, *, messages: Sequence[PromptMessage], max_token_limit: int, message_limit: int | None
    ) -> None:
        history = list(self._load_history())
        history.extend(messages)
        pruned = self._prune_prompt_messages(
            model_instance=self.model_instance,
            prompt_messages=history,
            max_token_limit=max_token_limit,
            message_limit=message_limit,
        )
        self._history = pruned
        context_ref = self._persist_context(pruned)
        if context_ref:
            self.metadata[WorkflowNodeExecutionMetadataKey.CONTEXT_REF] = context_ref

    def get_history_prompt_messages(
        self, max_token_limit: int = 2000, message_limit: int | None = None
    ) -> Sequence[PromptMessage]:
        history = list(self._load_history())
        if not history:
            return []

        return self._prune_prompt_messages(
            model_instance=self.model_instance,
            prompt_messages=history,
            max_token_limit=max_token_limit,
            message_limit=message_limit,
        )

    def run(
        self,
        model_parameters: dict[str, Any],
        output_schema: dict[str, Any],
        stop: list[str],
        user_id: str,
    ):
        from core.workflow.nodes.llm.exc import ModelNotExistError

        model_schema = self.model_instance.model_type_instance.get_model_schema(
            self.model_instance.model, self.model_instance.credentials
        )
        if model_schema is None:
            raise ModelNotExistError(f"Model schema not found for {self.model_instance.model}")
        return invoke_llm_with_structured_output(
            provider=self.model_instance.provider,
            model_schema=model_schema,
            model_instance=self.model_instance,
            prompt_messages=self.get_history_prompt_messages(max_token_limit=2000, message_limit=None),
            json_schema=output_schema,
            model_parameters=model_parameters,
            stop=stop,
            stream=False,
            user=user_id,
        )
