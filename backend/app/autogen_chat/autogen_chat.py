import autogen
from autogen_chat.user_proxy_webagent import UserProxyWebAgent
import asyncio
from collections import defaultdict
import datetime
from rag.rag_pipeline import initialize_rag_pipeline, query_rag_pipeline
import logging

# Configuration for the Large Language Models (LLMs) to be used in the chat
config_list = [
    {
        "model": "gpt-4o",  # Specifies the model version to be used
    }
]

# Configuration for the assistant agent using a LLM
llm_config_assistant = {
    "model": "gpt-3.5-turbo",  # Specifies the model version for the assistant agent
    "temperature": 0,  # Sets the creativity of the model's responses. 0 for deterministic.
    "config_list": config_list,  # References the LLM configuration defined above
    "functions": [
        {
            "name": "execute_rag_query",  # Function name to execute RAG queries
            "description": "Execute a RAG query and return the answer given the query and context retrieved",
            "parameters": {
                "type": "object",
                "properties": {
                    "rag_query": {
                        "type": "string",
                        "description": "The query to be executed by the RAG system.",
                    },
                },
                "required": ["rag_query"],  # 'rag_query' parameter is required
            },
        },
    ],
}

class AutogenChat():
    """
    Represents a chat session with an autogen assistant and a user proxy.

    Attributes:
        chat_id (str): Unique identifier for the chat session.
        websocket (WebSocket): WebSocket connection for real-time communication.
        client_sent_queue (asyncio.Queue): Queue for messages sent by the client.
        client_receive_queue (asyncio.Queue): Queue for messages to be received by the client.
        assistant (autogen.AssistantAgent): Autogen assistant agent for handling responses.
        user_proxy (UserProxyWebAgent): Proxy agent representing the user.
        rag_chain (RAGPipeline): Initialized RAG pipeline for executing queries.
    """
    def __init__(self, chat_id=None, websocket=None):
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = asyncio.Queue()
        self.client_receive_queue = asyncio.Queue()

        # Initialize the assistant agent with specific configuration and system message
        self.assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config_assistant,
            system_message="""You are a helpful legal assistant. Use legal language.
            
            Do not say things like: "Sure, I can help with that." or "I will now execute the RAG query.".
            First execute the RAG with what the user wants to get the context. If the user wants something, just execute the RAG query.
            Whenever you get the answer, do not use extra formatting like asterisks for bold or italics. Just provide the answer.
            """
        )

        # Initialize the user proxy agent with specific configurations
        self.user_proxy = UserProxyWebAgent(  
            name="user_proxy",
            human_input_mode="ALWAYS", 
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
            function_map={
                "execute_rag_query": self.execute_rag_query  # Maps 'execute_rag_query' to the corresponding method
            }
        )

        # Set queues for communication between the user proxy and the assistant
        self.user_proxy.set_queues(self.client_sent_queue, self.client_receive_queue)
        
        try:
            # Attempt to initialize the RAG pipeline
            self.rag_chain = initialize_rag_pipeline()
        except Exception as e:
            logging.error(f"Failed to initialize RAG pipeline: {e}")
            self.rag_chain = None

    async def start(self, message):
        """
        Starts the chat session by initiating communication between the user proxy and the assistant.

        Args:
            message (str): Initial message from the client to start the chat.
        """
        if not self.rag_chain:
            logging.error("RAG pipeline not initialized. Cannot start chat.")
            return

        try:
            await self.user_proxy.a_initiate_chat(
                self.assistant,
                clear_history=True,
                message=message
            )
        except Exception as e:
            logging.error(f"Failed to initiate chat: {e}")

    def execute_rag_query(self, rag_query):
        """
        Executes a RAG query using the initialized RAG pipeline.

        Args:
            rag_query (str): The query to be executed.

        Returns:
            str: The answer to the query or an error message if the execution fails.
        """
        if not self.rag_chain:
            logging.error("RAG pipeline not initialized. Cannot execute query.")
            return "Error: RAG pipeline not available."

        try:
            print("--EXECUTING_RAG-- RAG QUERY")
            print("--INFO-- MESSAGE", rag_query)

            answer = query_rag_pipeline(self.rag_chain, rag_query)
            return answer + " TERMINATE"
        except Exception as e:
            logging.error(f"Failed to execute RAG query: {e}")
            return "Error: Failed to execute RAG query."