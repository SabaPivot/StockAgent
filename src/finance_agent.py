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
    def __init__(self, today: str = today, **kwargs):
        super().__init__(name="finance_toolkit")
        
        # Initialize the underlying finance agent with the desired configuration.
        self.finance_agent = Agent(
            name="Finance Agent",
            role="Search the web to provide Financial guide.",
            model=OpenAIChat(id="gpt-4o", max_tokens=1024, temperature=0.1),
            tools=[
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    company_info=True
                )
            ],
            description=dedent("""\
                You are a turly sincere financial assistant that answers every question.
                You are news-sensitive. So reach out to ExaTools to fetch the latest news.
                Always state your source.
            """),
            instructions=dedent("""\
                You are a skilled financial analyst with expertise in market data!
                Always answer with grounding facts and evidence. **Do not speak generally**.
                                
                You must follow the format below:

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

        self.web_agent=Agent(
            name="Web Agent",
            role="Search the web for information",
            model=OpenAIChat(id="gpt-4o"),
            tools=[ExaTools()],
            instructions=dedent("""\
                You are an experienced web researcher and news analyst! 🔍

                Follow these steps when searching for information:
                1. Start with the most recent and relevant sources
                2. Cross-reference information from multiple sources
                3. Prioritize reputable news outlets and official sources
                4. Always cite your sources with links
                5. Focus on market-moving news and significant developments

                Your style guide:
                - Present information in a clear, journalistic style
                - Use bullet points for key takeaways
                - Include relevant quotes when available
                - Specify the date and time for each piece of news
                - Highlight market sentiment and industry trends
                - End with a brief analysis of the overall narrative
                - Pay special attention to regulatory news, earnings reports, and strategic announcements\
            """),
            show_tool_calls=True,
            markdown=True,
        )
        self.agent_team = Agent(
            team=[self.web_agent, self.finance_agent],
            model=OpenAIChat(id="gpt-4o"),
            instructions=dedent("""\
                You are the lead editor of a prestigious financial news desk! 📰

                Your role:
                1. Coordinate between the web researcher and financial analyst
                2. Combine their findings into a compelling narrative
                3. Ensure all information is properly sourced and verified
                4. Present a balanced view of both news and data
                5. Highlight key risks and opportunities

                Your style guide:
                - Start with an attention-grabbing headline
                - Begin with a powerful executive summary
                - Present financial data first, followed by news context
                - Use clear section breaks between different types of information
                - Include relevant charts or tables when available
                - Add 'Market Sentiment' section with current mood
                - Include a 'Key Takeaways' section at the end
                - End with 'Risk Factors' when appropriate
                - Sign off with 'Market Watch Team' and the current date\
            """),
            add_datetime_to_instructions=True,
            show_tool_calls=True,
            markdown=True,
        )
        
        # Register the ask_finance_agent method in the toolkit.
        self.register(self.ask_finance_agent_team)

    def ask_finance_agent_team(self, query: str) -> str:
        """
        Runs the finance agent with the provided query and returns its response.
        """
        try:
            logger.info(f"Running Agent team with query: {query}")
            result = self.agent_team.run(query)
            logger.info("Agent team response received.")
            return result.content
        except Exception as e:
            logger.error(f"Error running Finance Agent: {e}")
            return f"Error: {e}"

# class FinanceToolkit(Toolkit):
#     """
#     FinanceToolkit directly integrates financial data tools and provides a method
#     to analyze financial queries without wrapping an Agent.
#     """
#     def __init__(self, today: str = today, **kwargs):
#         super().__init__(name="finance_toolkit")
        
#         # Instantiate the underlying tools directly.
#         self.yfinance = YFinanceTools(
#             stock_price=True,
#             analyst_recommendations=True,
#             company_info=True
#         )
#         self.exa = ExaTools(start_published_date=today, type="keyword")
        
#         # Register the main method for finance analysis.
#         self.register(self.analyze_finance)

#     def analyze_finance(self, query: str) -> str:
#         """
#         Processes the financial query by directly calling the underlying tools.
#         Combines stock data, analyst recommendations, and additional market info.
#         """
#         try:
#             logger.info(f"Processing finance query: {query}")
            
#             # Retrieve financial data using YFinanceTools.
#             # (Replace these method calls with the actual methods your tool provides.)
#             stock_data = self.yfinance.get_current_stock_price(query)
#             analyst_data = self.yfinance.get_analyst_recommendations(query)
#             company_info = self.yfinance.get_company_info(query)
#             stock_fundamentals = self.yfinance.get_stock_fundamentals(query)
            
#             # Retrieve additional market info using ExaTools.
#             # (Replace this with the appropriate search method from ExaTools.)
#             extra_info = self.exa.search_exa(query)
            
#             # Compose a final report. You can adjust formatting (tables, markdown, etc.)
#             report = dedent(f"""
#             ### Financial Analysis Report for {query}
            
#             **Stock Price Data:**  
#             {stock_data}

#             **Stock Fudamental Data:**
#             {stock_fundamentals}
            
#             **Analyst Recommendations:**  
#             {analyst_data}
            
#             **Company Info:**  
#             {company_info}
            
#             **Additional Market Information:**  
#             {extra_info}
#             """)
            
#             logger.info("Finance analysis completed successfully.")
#             return report
#         except Exception as e:
#             logger.error(f"Error analyzing finance data: {e}")
#             return f"Error: {e}"