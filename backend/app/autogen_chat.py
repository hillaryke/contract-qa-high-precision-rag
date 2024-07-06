import autogen
from user_proxy_webagent import UserProxyWebAgent
import asyncio
from collections import defaultdict
import datetime
timer_array = []

config_list = [
    {
        "model": "gpt-4o",
    }
]
llm_config_assistant = {
    "model":"gpt-4o",
    "temperature": 0,
    "config_list": config_list,
        "functions": [
        {
            "name": "execute_rag_query",
            "description": "Execute a RAG query and return the answer given the query and context retrieved",
            "parameters": {
                "type": "object",
                "properties": {
                    "rag_query": {
                        "type": "string",
                        "description": "The query to be executed by the RAG system.",
                    },
                },
                "required": ["rag_query"],
            },
        },
    ],
}

#############################################################################################
# this is where you put your Autogen logic, here I have a simple 2 agents with a function call
class AutogenChat():
    def __init__(self, chat_id=None, websocket=None):
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = asyncio.Queue()
        self.client_receive_queue = asyncio.Queue()

        self.assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config_assistant,
            system_message="""You are a helpful legal assistant. Use legal language 
            
            Do not say things like: "Sure, I can help with that." or "I will now execute the RAG query.".
            First execute the RAG with the what the user wants to get the context. If the use wants something just execute the RAG query.
            """
        )
        self.user_proxy = UserProxyWebAgent(  
            name="user_proxy",
            human_input_mode="ALWAYS", 
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
            function_map={
                "execute_rag_query": self.execute_rag_query
            }
        )

        # add the queues to communicate 
        self.user_proxy.set_queues(self.client_sent_queue, self.client_receive_queue)

    async def start(self, message):
        await self.user_proxy.a_initiate_chat(
            self.assistant,
            clear_history=True,
            message=message
        )
        
    def execute_rag_query(self, rag_query):
        print("--INFO-- RAG QUERY")
        print("--INFO-- MESSAGE", rag_query)
        # if message.startswith("RAG_QUERY:"):
            # Extract the query and get the answer from the RAG pipeline
            # query = message[len("RAG_QUERY:"):].strip()
            # print("--INFO-- QUERY", query)
            # response = self.rag_chain.invoke({"question": query})
            # Mock response
        response = {"answer": "According to section 3.2 of the contract, the delivery date is 15th June 2022."}
        answer = response["answer"]
        return answer + " TERMINATE"
        # else:
        #     # If not a RAG query, pass the message to the assistant
        #     await self.client_sent_queue.put(message)
        #     return None  # No direct response needed
