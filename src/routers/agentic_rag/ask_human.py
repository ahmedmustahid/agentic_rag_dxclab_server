import json

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from langgraph.types import StreamWriter
from typing_extensions import Annotated

from src.routers.agentic_rag.message_utils import MsgUtils
from src.routers.agentic_rag.param_llm import get_gpt_model
from src.routers.agentic_rag.state import State
from src.routers.utils.agent_msg_manager import AgentMsgManager
from src.routers.utils.log_dev import LogDev
from src.routers.utils.prompt_manager import PromptManager

log = LogDev()
load_dotenv()

model = get_gpt_model()
msg_util = MsgUtils()
prompt_mgr = PromptManager()
agent_msg_mgr = AgentMsgManager()


class AskHumanAgent:
    """
    AskHumanAgent
    Ask user the question
    """

    def __init__(self):
        pass

    def ask_human(
        self,
        state: Annotated[State, InjectedState],
        config: RunnableConfig,
        writer: StreamWriter,
    ) -> dict:
        """
        Ask user the question

        Args:
          state: State:

        Returns:
          str: answer
        """

        # Prompt
        prompt_template = prompt_mgr.get_prompt("ask_human")
        messages = state["messages"]
        # Latest Question and Request from Users (Updated)
        rev_request = state["rev_request"]
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["rev_request", "msg_history"]
        )
        msg_history = msg_util.get_pure_msg(messages)
        rev_request = state["rev_request"]
        input_data = {"rev_request": rev_request, "msg_history": msg_history}
        chain_llm = prompt | model | StrOutputParser()
        answer_txt = chain_llm.invoke(input_data, config=config)
        answer = AIMessage(
            content=json.dumps(answer_txt, ensure_ascii=False), additional_kwargs={}
        )
        # Show logs
        msg = agent_msg_mgr.get_msg("ask_human", content=answer_txt)
        log.print(msg + "\n")
        # Streaming custom message
        writer(msg)
        return {"messages": [answer]}
