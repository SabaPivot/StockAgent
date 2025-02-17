from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from rich import print
import typer
from typing import List
from dataclasses import dataclass
from textwrap import dedent
from finance_agent import FinanceToolkit

@dataclass
class ChatbotAgent:
    user: str = "Samuel"
    agent_storage: SqliteAgentStorage = SqliteAgentStorage(table_name="chatbot", db_file="tmp/chat.db")
    agent = None
    session_id: str = None

    def if_new_session(self) -> bool:
        """Ask the user if they want to start a new session."""
        return typer.confirm("Do you want to start a new session?")

    def get_session_id(self) -> str:
        """Retrieve the latest session ID or start a new session."""
        session_id = None

        if not self.if_new_session():
            existing_sessions: List[str] = self.agent_storage.get_all_session_ids()
            if existing_sessions:
                session_id = existing_sessions[0]  # Get the latest session
        return session_id

    def create_agent(self):
        """Initialize the chatbot with financial capabilities."""
        self.session_id = self.get_session_id()

        self.agent = Agent(
            name="Chatbot",
            role="A chatbot assisting the user with everything.",
            model=OpenAIChat(id="gpt-4o", max_tokens=512, temperature=0.1),
            show_tool_calls=True,
            add_datetime_to_instructions=True,
            session_id=self.session_id,
            markdown=True,
            knowledge=None,
            storage=self.agent_storage,
            read_chat_history=True,
            add_history_to_messages=True,
            tools=[FinanceToolkit()],
            instructions="Use tools to fetch a real time data",
        )

        if self.session_id is None:
            print("📌 Starting a new session\n")
        else:
            print(f"📌 Continuing session {self.session_id}\n")

    def launch(self):
        self.agent.cli_app(markdown=True)
        session_ids = self.agent_storage.get_all_session_ids()
        print(f"✅ Session {session_ids[0]} is closed.")