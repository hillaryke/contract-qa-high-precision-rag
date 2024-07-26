from fastapi import FastAPI, WebSocket
from autogen_chat import AutogenChat
import asyncio
import uvicorn
from dotenv import load_dotenv, find_dotenv
import openai
import os

_ = load_dotenv(find_dotenv()) # Load environment variables from .env file
openai.api_key = os.environ['OPENAI_API_KEY']  # Set OpenAI API key from environment variable

app = FastAPI()  # Create a FastAPI application instance
app.autogen_chat = {}  # Initialize a dictionary to store chat instances

class ConnectionManager:
    """
    Manages connections for the chat application.
    
    Attributes:
        active_connections (list[AutogenChat]): A list to keep track of active chat connections.
    """
    def __init__(self):
        self.active_connections: list[AutogenChat] = []

    async def connect(self, autogen_chat: AutogenChat):
        """
        Accepts a websocket connection and adds it to the list of active connections.
        
        Args:
            autogen_chat (AutogenChat): The chat instance to connect.
            
        Returns:
            None
        """
        await autogen_chat.websocket.accept()
        self.active_connections.append(autogen_chat)

    async def disconnect(self, autogen_chat: AutogenChat):
        """
        Removes a chat instance from the list of active connections and signals it to finish.
        
        Args:
            autogen_chat (AutogenChat): The chat instance to disconnect.
            
        Returns:
            None
        """
        autogen_chat.client_receive_queue.put_nowait("DO_FINISH")
        print(f"autogen_chat {autogen_chat.chat_id} disconnected")
        self.active_connections.remove(autogen_chat)

manager = ConnectionManager()  # Create an instance of ConnectionManager

async def send_to_client(autogen_chat: AutogenChat):
    """
    Continuously sends messages from the server to the client.
    
    Args:
        autogen_chat (AutogenChat): The chat instance through which messages are sent.
        
    Returns:
        None
    """
    print("--INFO-- SEND_TO_CLIENT")
    while True:
        reply = await autogen_chat.client_receive_queue.get()
        if reply and reply == "DO_FINISH":
            autogen_chat.client_receive_queue.task_done()
            break
        await autogen_chat.websocket.send_text(reply)
        autogen_chat.client_receive_queue.task_done()
        await asyncio.sleep(0.05)

async def receive_from_client(autogen_chat: AutogenChat):
    """
    Continuously receives messages from the client.
    
    Args:
        autogen_chat (AutogenChat): The chat instance through which messages are received.
        
    Returns:
        None
    """
    print("--INFO-- RECEIVE_FROM_CLIENT")
    while True:
        data = await autogen_chat.websocket.receive_text()
        if data and data == "DO_FINISH":
            await autogen_chat.client_receive_queue.put("DO_FINISH")
            await autogen_chat.client_sent_queue.put("DO_FINISH")
            break
        await autogen_chat.client_sent_queue.put(data)
        await asyncio.sleep(0.05)

@app.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    """
    WebSocket endpoint for handling chat connections.
    
    Args:
        websocket (WebSocket): The WebSocket connection.
        chat_id (str): Unique identifier for the chat session.
        
    Returns:
        None
    """
    try:
        autogen_chat = AutogenChat(chat_id=chat_id, websocket=websocket)
        print("--INFO-- CONNECT")
        await manager.connect(autogen_chat)
        data = await autogen_chat.websocket.receive_text()
        print("--INFO-- START")
        future_calls = asyncio.gather(send_to_client(autogen_chat), receive_from_client(autogen_chat))
        print("--INFO-- AFTER FUTURE CALLS")
        await autogen_chat.start(data)
        print("DO_FINISHED")
    except Exception as e:
        print("ERROR", str(e))
    finally:
        try:
            await manager.disconnect(autogen_chat)
        except:
            pass

# For Production
# if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)