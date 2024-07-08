import autogen
from autogen_chat.user_proxy_webagent import UserProxyWebAgent
import asyncio
from collections import defaultdict
import datetime
from rag.rag_pipeline import initialize_rag_pipeline, query_rag_pipeline
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb
from autogen.agentchat.contrib.retrieve_user_proxy_agent import PROMPT_QA

timer_array = []

config_list = [
    {
        "model": "gpt-4o",
    }
]
llm_config_assistant = {
    "model":"gpt-4o",
    "temperature": 0,
    "config_list": config_list
}


#############################################################################################
# this is where we place our Autogen logic, here we have simple 2 agents, the assistant and the user_proxy
class AutogenChat():
    """
    A class to manage chat interactions between an assistant and a user through websockets.
    
    This class initializes chat components, including the assistant agent and user proxy agent,
    and manages message queues for sending and receiving messages between the client and the server.
    
    Attributes:
        websocket (Websocket): The websocket connection for real-time communication.
        chat_id (str): Unique identifier for the chat session.
        client_sent_queue (asyncio.Queue): Queue for messages sent from the client to the server.
        client_receive_queue (asyncio.Queue): Queue for messages to be received by the client from the server.
        assistant (AssistantAgent): The assistant agent handling automated responses.
        user_proxy (UserProxyWebAgent): The user proxy agent managing user interactions.
    
    Methods:
        start(message): Initiates the chat with a user message and prepares the assistant response.
    """
    
    def __init__(self, chat_id=None, websocket=None):
        """
        Initializes the AutogenChat class with a chat ID, websocket, and sets up the assistant and user proxy agents.
        
        Parameters:
            chat_id (str, optional): Unique identifier for the chat session. Defaults to None.
            websocket (Websocket, optional): The websocket connection for real-time communication. Defaults to None.
        
        Returns:
            None
        """
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = asyncio.Queue()
        self.client_receive_queue = asyncio.Queue()

        self.assistant = autogen.AssistantAgent(
            name="assistant",
            system_message="""You are a helpful assistant."""
        )
        
        self.user_proxy = UserProxyWebAgent(  
            name="ragproxyagent",
            human_input_mode="ALWAYS", 
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            retrieve_config={
                "task": "qa",
                "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
                "chunk_token_size": 2000,
                "model": config_list[0]["model"],
                "client": chromadb.PersistentClient(path="/tmp/chromadb"),
                "embedding_model": "all-mpnet-base-v2",
                "customized_prompt": PROMPT_QA,
                "get_or_create": True,
                "collection_name": "autogen_rag",
            },
            code_execution_config=False,
        )

        self.user_proxy.set_queues(self.client_sent_queue, self.client_receive_queue)
        
    async def start(self, message):
        """
        Initiates the chat by sending a user message to the assistant and setting up the response mechanism.
        
        Parameters:
            message (str): The initial message from the user to start the chat.
        
        Returns:
            None
        """
        await self.user_proxy.a_initiate_chat(
            self.assistant,
            clear_history=True,
            message=self.user_proxy.message_generator,
            problem=message
        )