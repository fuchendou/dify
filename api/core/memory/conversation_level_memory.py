from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from core.app.app_config.features.file_upload.manager import FileUploadConfigManager
from core.file import file_manager
from core.memory.base import BufferMemory
from core.model_manager import ModelInstance
from core.model_runtime.entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)
from core.model_runtime.entities.message_entities import PromptMessageContentUnionTypes
from core.prompt.utils.extract_thread_messages import extract_thread_messages
from extensions.ext_database import db
from factories import file_factory
from models.model import AppMode, Conversation, Message, MessageFile
from models.workflow import Workflow
from repositories.api_workflow_run_repository import APIWorkflowRunRepository
from repositories.factory import DifyAPIRepositoryFactory


class ConversationLevelMemory(BufferMemory):
    def __init__(
        self,
        conversation: Conversation,
        model_instance: ModelInstance,
    ):
        super().__init__(model_instance=model_instance)
        self.conversation = conversation
        self._workflow_run_repo: APIWorkflowRunRepository | None = None

    @property
    def workflow_run_repo(self) -> APIWorkflowRunRepository:
        if self._workflow_run_repo is None:
            session_maker = sessionmaker(bind=db.engine, expire_on_commit=False)
            self._workflow_run_repo = DifyAPIRepositoryFactory.create_api_workflow_run_repository(session_maker)
        return self._workflow_run_repo

    def _build_prompt_message_with_files(
        self,
        message_files: Sequence[MessageFile],
        text_content: str,
        message: Message,
        app_record,
        is_user_message: bool,
    ) -> PromptMessage:
        if self.conversation.mode in {AppMode.AGENT_CHAT, AppMode.COMPLETION, AppMode.CHAT}:
            file_extra_config = FileUploadConfigManager.convert(self.conversation.model_config)
        elif self.conversation.mode in {AppMode.ADVANCED_CHAT, AppMode.WORKFLOW}:
            app = self.conversation.app
            if not app:
                raise ValueError("App not found for conversation")

            if not message.workflow_run_id:
                raise ValueError("Workflow run ID not found")

            workflow_run = self.workflow_run_repo.get_workflow_run_by_id(
                tenant_id=app.tenant_id, app_id=app.id, run_id=message.workflow_run_id
            )
            if not workflow_run:
                raise ValueError(f"Workflow run not found: {message.workflow_run_id}")
            workflow = db.session.scalar(select(Workflow).where(Workflow.id == workflow_run.workflow_id))
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_run.workflow_id}")
            file_extra_config = FileUploadConfigManager.convert(workflow.features_dict, is_vision=False)
        else:
            raise AssertionError(f"Invalid app mode: {self.conversation.mode}")

        detail = ImagePromptMessageContent.DETAIL.HIGH
        if file_extra_config and app_record:
            file_objs = [
                file_factory.build_from_message_file(
                    message_file=message_file, tenant_id=app_record.tenant_id, config=file_extra_config
                )
                for message_file in message_files
            ]
            if file_extra_config.image_config and file_extra_config.image_config.detail:
                detail = file_extra_config.image_config.detail
        else:
            file_objs = []

        if not file_objs:
            return (
                UserPromptMessage(content=text_content)
                if is_user_message
                else AssistantPromptMessage(content=text_content)
            )

        prompt_message_contents: list[PromptMessageContentUnionTypes] = []
        for file in file_objs:
            prompt_message = file_manager.to_prompt_message_content(
                file,
                image_detail_config=detail,
            )
            prompt_message_contents.append(prompt_message)
        prompt_message_contents.append(TextPromptMessageContent(data=text_content))

        return (
            UserPromptMessage(content=prompt_message_contents)
            if is_user_message
            else AssistantPromptMessage(content=prompt_message_contents)
        )

    def get_history_prompt_messages(
        self, max_token_limit: int = 2000, message_limit: int | None = None
    ) -> Sequence[PromptMessage]:
        app_record = self.conversation.app

        stmt = (
            select(Message).where(Message.conversation_id == self.conversation.id).order_by(Message.created_at.desc())
        )

        if message_limit and message_limit > 0:
            message_limit = min(message_limit, 500)
        else:
            message_limit = 500

        msg_limit_stmt = stmt.limit(message_limit)
        messages = db.session.scalars(msg_limit_stmt).all()

        thread_messages = extract_thread_messages(messages)
        if thread_messages and not thread_messages[0].answer and thread_messages[0].answer_tokens == 0:
            thread_messages.pop(0)

        messages = list(reversed(thread_messages))

        prompt_messages: list[PromptMessage] = []
        for message in messages:
            user_files = db.session.scalars(
                select(MessageFile).where(
                    MessageFile.message_id == message.id,
                    (MessageFile.belongs_to == "user") | (MessageFile.belongs_to.is_(None)),
                )
            ).all()

            if user_files:
                user_prompt_message = self._build_prompt_message_with_files(
                    message_files=user_files,
                    text_content=message.query,
                    message=message,
                    app_record=app_record,
                    is_user_message=True,
                )
                prompt_messages.append(user_prompt_message)
            else:
                prompt_messages.append(UserPromptMessage(content=message.query))

            assistant_files = db.session.scalars(
                select(MessageFile).where(MessageFile.message_id == message.id, MessageFile.belongs_to == "assistant")
            ).all()

            if assistant_files:
                assistant_prompt_message = self._build_prompt_message_with_files(
                    message_files=assistant_files,
                    text_content=message.answer,
                    message=message,
                    app_record=app_record,
                    is_user_message=False,
                )
                prompt_messages.append(assistant_prompt_message)
            else:
                prompt_messages.append(AssistantPromptMessage(content=message.answer))

        if not prompt_messages:
            return []

        return self._prune_prompt_messages(
            model_instance=self.model_instance,
            prompt_messages=prompt_messages,
            max_token_limit=max_token_limit,
            message_limit=message_limit,
        )
