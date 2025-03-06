from chatbot import ChatbotAgent
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

if __name__ == "__main__":
    agent = ChatbotAgent()
    agent.create_agent()
    agent.launch()
