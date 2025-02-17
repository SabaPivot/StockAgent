from agno.tools.yfinance import YFinanceTools
from agno.tools.exa import ExaTools
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from textwrap import dedent
from agno.utils.log import logger
from datetime import date
from agno.tools import Toolkit

today = date.today().strftime("%Y-%m-%d")

class FinanceToolkit(Toolkit):
    """
    FinanceToolkit is a toolkit for providing financial insights by leveraging an underlying finance agent.
    It registers the `ask_finance_agent` method, which uses the finance agent to process queries.
    """
    def __init__(self, today: str = today, **kwargs):
        super().__init__(name="finance_toolkit")
        
        # Initialize the underlying finance agent with the desired configuration.
        self.finance_agent = Agent(
            name="Finance Agent",
            role="Search the web to provide Financial guide.",
            model=OpenAIChat(id="gpt-4o", max_tokens=512, temperature=0.1),
            tools=[
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    company_info=True
                ),
                ExaTools(start_published_date=today, type="keyword")
            ],
            instructions=dedent("""\
                You are a skilled financial analyst with expertise in market data!

                Follow these steps when analyzing financial data:
                1. Start with the latest stock price, trading volume, and daily range.
                2. Present detailed analyst recommendations and consensus target prices.
                3. Include key metrics.
                4. Analyze trading patterns and volume trends.
                5. Compare performance against relevant sector indices.
                6. Must provide Fear&Greedy Index of the market. (Search on web)

                Your style guide:
                - Use tables for structured data presentation.
                - Include clear headers for each data section.
                - Add brief explanations for technical terms.
                - Highlight notable changes with emojis (📈 📉).
                - Use bullet points for quick insights.
                - Compare current values with historical averages.
                - End with a data-driven financial outlook.
            """),
            show_tool_calls=True,
            markdown=True,
            add_datetime_to_instructions=True,
        )
        
        # Register the ask_finance_agent method in the toolkit.
        self.register(self.ask_finance_agent)

    def ask_finance_agent(self, query: str) -> str:
        """
        Runs the finance agent with the provided query and returns its response.
        """
        try:
            logger.info(f"Running Finance Agent with query: {query}")
            result = self.finance_agent.run(query)
            logger.info("Finance Agent response received.")
            return result.content
        except Exception as e:
            logger.error(f"Error running Finance Agent: {e}")
            return f"Error: {e}"