from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from rich import print
import typer
from typing import List
from dataclasses import dataclass
from textwrap import dedent
from finance_agent import FinanceToolkit
from agno.tools.duckduckgo import DuckDuckGoTools

@dataclass
class ChatbotAgent:
    user: str = "Samuel"
    agent_storage: SqliteAgentStorage = SqliteAgentStorage(table_name="chatbot", db_file="tmp/chat.db")
    agent = None
    session_id: str = None

    def if_new_session(self) -> bool:
        return typer.confirm("Do you want to start a new session?")

    def get_session_id(self) -> str:
        session_id = None

        if not self.if_new_session():
            existing_sessions: List[str] = self.agent_storage.get_all_session_ids()
            if existing_sessions:
                session_id = existing_sessions[0]  # Get the latest session
        return session_id

    def create_agent(self):
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
            tools=[FinanceToolkit(), DuckDuckGoTools()],
            description=dedent("""\
                You are an helpful and nice assistant that answers almost every questions.
                Never answer with 'sorry', 'I cannot fetch real-time data'.
            """),
            instructions=dedent("""\
                When you are asked about the following topics:
                *1. companies*
                *2. economy*
                *3. stocks*
                *4. investment*
                *5. interest rate*
                *6. latest news about economy and finance*
                Always find FinanceToolkit for help. Your answers on these topic should be organized and have grounding factors.
                
                Otherwise, you can look for DuckDuckGoTools for latest information.
                For instance, if you are asked about weather, you can find it via DuckDuckGoTools.

                Your answer must be always organized.
            """),
        )

        if self.session_id is None:
            print("📌 Starting a new session\n")
        else:
            print(f"📌 Continuing session {self.session_id}\n")

    def launch(self):
        self.agent.cli_app(markdown=True)
        session_ids = self.agent_storage.get_all_session_ids()
        print(f"✅ Session {session_ids[0]} is closed.")