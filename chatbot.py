from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.vectordb.lancedb import LanceDb, SearchType
from rich import print
import typer
from typing import List
from dataclasses import dataclass

@dataclass
class ChatbotAgent:
    user: str = "Samuel"
    agent_storage: SqliteAgentStorage = SqliteAgentStorage(table_name="chatbot", db_file="tmp/chat.db")
    new_session: bool = None
    agent = None


    def if_new_session(self) -> bool:
        return typer.confirm("Do you want to start a new session?")

    def get_session_id(self) -> str:
        session_id = None  # ✅ Ensure session_id is explicitly initialized

        if not self.if_new_session():  # ✅ Uses the returned boolean
            existing_sessions: List[str] = self.agent_storage.get_all_session_ids()
            if existing_sessions:
                session_id = existing_sessions[0]  # Get the latest session
        return session_id

    def create_agent(self):
        session_id = self.get_session_id()
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o", max_tokens=512, temperature=0.1),
            session_id=session_id,
            description="Answer the question in a sentence.",
            markdown=True,
            knowledge=None,
            storage=self.agent_storage,
            read_chat_history=True,
            add_history_to_messages=True,
        )
        
        if session_id is None:
            print(f"Starting new session\n")
        else:
            print(f"Continuing session {session_id}\n")

    def run(self):
        self.agent.cli_app(markdown=True)
        print(f"Session {self.agent_storage.get_all_session_ids()[0]} is closed.")
