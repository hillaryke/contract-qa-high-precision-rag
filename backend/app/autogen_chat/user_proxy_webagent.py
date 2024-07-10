import autogen
from autogen import Agent, ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import asyncio
import logging

try:
    from termcolor import colored
except ImportError:
    # Provides a fallback for the colored function if termcolor is not installed.
    def colored(x, *args, **kwargs):
        return x

class UserProxyWebAgent(autogen.UserProxyAgent):
    """
    A class that represents a user proxy web agent, extending functionality from autogen.UserProxyAgent.
    This agent is designed to interact with users, handling messages and determining whether to continue
    conversations based on user input and predefined conditions.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the UserProxyWebAgent with the given arguments.
        It also initializes a list to keep track of reply functions and registers several default reply handlers.
        """
        super(UserProxyWebAgent, self).__init__(*args, **kwargs)
        self._reply_func_list = []  # List to store reply functions.
        # Register default reply handlers for various scenarios.
        self.register_reply([Agent, None], ConversableAgent.generate_oai_reply)
        self.register_reply([Agent, None], ConversableAgent.generate_code_execution_reply)
        self.register_reply([Agent, None], ConversableAgent.generate_function_call_reply)
        self.register_reply([Agent, None], UserProxyWebAgent.a_check_termination_and_human_reply)

    async def a_check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """
        Asynchronously checks if the conversation should be terminated based on the latest message,
        and if a human reply is provided or required.

        Args:
            messages (Optional[List[Dict]]): The list of messages in the conversation.
            sender (Optional[Agent]): The agent sending the message.
            config (Optional[Any]): Configuration object, if any.

        Returns:
            Tuple[bool, Union[str, Dict, None]]: A tuple containing a boolean indicating if the conversation
            should stop, and the reply (str or Dict) or None if the conversation is to be terminated.
        """
        # Default configurations and message handling.
        if config is None:
            config = self
        if messages is None: 
            messages = self._oai_messages[sender]
        message = messages[-1]  # Get the last message.
        reply = ""
        no_human_input_msg = ""
        # Handling based on human input mode.
        if self.human_input_mode == "ALWAYS":
            # Always require human input.
            reply = await self.a_get_human_input(
                f"Provide feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: "
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            # Handling based on the number of consecutive auto-replies and termination conditions.
            if self._consecutive_auto_reply_counter[sender] >= self._max_consecutive_auto_reply_dict[sender]:
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    terminate = self._is_termination_msg(message)
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    reply = reply or "exit"

        # Print message if no human input was received.
        if no_human_input_msg:
            print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # Handling for stopping the conversation.
        if reply == "exit":
            self._consecutive_auto_reply_counter[sender] = 0  # Reset counter.
            return True, None

        # Send the human reply or continue with auto-reply.
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            self._consecutive_auto_reply_counter[sender] = 0  # Reset counter.
            return True, reply

        # Increment the counter for consecutive auto-replies.
        self._consecutive_auto_reply_counter[sender] += 1
        if self.human_input_mode != "NEVER":
            print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)

        return False, None

    def set_queues(self, client_sent_queue, client_receive_queue):
        """
        Sets the queues for sending and receiving messages to/from the client.

        Args:
            client_sent_queue: The queue for sending messages to the client.
            client_receive_queue: The queue for receiving messages from the client.
        """
        self.client_sent_queue = client_sent_queue
        self.client_receive_queue = client_receive_queue

    async def a_get_human_input(self, prompt: str) -> str:
        """
        Asynchronously gets human input based on the last message content.

        Args:
            prompt (str): The prompt to display for human input.

        Returns:
            str: The human input or 'exit' if the conversation is to be terminated.
        """
        try:
            last_message = self.last_message()
            if last_message and last_message.get("content"):
                await self.client_receive_queue.put(last_message["content"])
                reply = await self.client_sent_queue.get()
                if reply and reply == "DO_FINISH":
                    return "exit"
                return reply
            else:
                return ""
        except asyncio.QueueEmpty:
            logging.error("Queue is empty, failed to get human input.")
            return "exit"
        except Exception as e:
            logging.error(f"Unexpected error in a_get_human_input: {e}")
            return "exit"