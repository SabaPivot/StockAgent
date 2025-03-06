import sqlite3
from typing import List, Dict

def extract_conversation_by_session(db_path: str, session_id: str, table_name: str = "chatbot") -> List[Dict[str, str]]:
    """Extracts user & assistant messages filtered by a specific session_id.
    
    Assumes that the chatbot.memory column contains a JSON object with a key "runs", 
    where each run has a nested "response.messages" array. The query filters runs whose 
    response.session_id matches the provided session_id.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    messages_query = """
    SELECT 
        json_extract(json_each_messages.value, '$.content') AS message,
        json_extract(json_each_messages.value, '$.role') AS role
    FROM {table_name},
         json_each(json_extract(json(chatbot.memory), '$.runs')) AS json_each_runs,
         json_each(json_extract(json_each_runs.value, '$.response.messages')) AS json_each_messages
    WHERE json_extract(json_each_runs.value, '$.response.session_id') = ? 
      AND json_extract(json_each_messages.value, '$.role') IN ('user', 'assistant');
    """.format(table_name=table_name)

    cursor.execute(messages_query, (session_id,))
    messages = cursor.fetchall()
    conn.close()

    # Organize messages into a structured list
    conversation_history = [{"role": role, "message": message} for message, role in messages]

    return conversation_history
