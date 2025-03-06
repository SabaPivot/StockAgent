# StockAgent

A Streamlit-based financial chatbot powered by Agno and GPT-4 that provides comprehensive financial analysis, real-time market insights, and news-driven market intelligence.

## Features
### Financial Analysis
- Real-time stock price tracking and analysis
- Detailed analyst recommendations and target prices
- Company fundamental data analysis
- Trading pattern and volume trend analysis
- Sector performance comparisons
- Fear & Greed Index monitoring

### Market Intelligence
- Real-time financial news integration via Exa
- Multi-source news verification
- Market sentiment analysis
- Regulatory updates and earnings reports
- Industry trend analysis

### User Experience
- Interactive chat interface
- Persistent session-based conversations
- Structured data presentation with tables
- Clear section organization with headers
- Technical term explanations
- Visual indicators for market changes

## Dependencies
- agno==1.1.4
- openai==1.63.2
- SQLAlchemy==2.0.38
- yfinance==0.2.53
- exa-py==1.8.8
- duckduckgo_search==7.4.2
- python-dotenv
- streamlit

## Setup
1. Set up environment variables:
Create a `.env` file in the root directory with:
```
EXA_API_KEY=your_exa_api_key
OPENAI_API_KEY=your_openai_api_key
```

2. Install dependencies:
```bash
pip install -r src/requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## Project Structure
- `src/`: Source code directory containing core functionality
  - `chatbot.py`: Core chatbot implementation using GPT-4 with:
    - Financial toolkit integration
    - DuckDuckGo search capabilities
    - Session management
    - SQLite storage for chat history
  - `finance_agent.py`: Advanced financial analysis system with:
    - Real-time stock data via YFinance
    - News integration via Exa
    - Multi-agent architecture (Web Agent + Finance Agent)
    - Structured analysis templates
    - Market sentiment analysis
  - `main.py`: Application entry point with:
    - Environment configuration
    - Agent initialization
    - CLI interface
  - `message_history.py`: Chat history management with:
    - SQLite-based storage
    - Session-based message retrieval
    - JSON-based message structure
- `streamlit_app.py`: Streamlit web interface with:
  - Session management
  - Chat history display
  - Real-time chat interface

## Storage
The application uses SQLite (`tmp/chat.db`) for storing:
- Chat history in JSON format
- Session information
- User and assistant messages
- Run metadata

## License
MIT License
