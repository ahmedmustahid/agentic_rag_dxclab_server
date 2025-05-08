import datetime
import json

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
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


class AnswerLlmAgent:
    """
    AnswerLlmAgent
    Answer user requests using only LLM knowledge
    """

    def __init__(self):
        pass

    def ans_llm_solo(
        self, state: Annotated[State, InjectedState], config: RunnableConfig
    ) -> dict:
        """
        Answers based on LLM's base knowledge, called directly by the parent Agent

        Args:
          state: State:

        Returns:
          str: answer
        """
        log.print("- Start: ans_llm_solo")
        messages = state["messages"]
        # Prompt
        prompt_template = prompt_mgr.get_prompt("ans_llm_solo")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["request", "msg_history", "date_time"],
        )
        msg_history = msg_util.get_pure_msg(messages)
        # Latest Question and Request from Users (Updated)
        rev_request = state["rev_request"]
        # Get current date and time
        now = datetime.datetime.now()
        # Specify the date and time format (e.g. 2025-03-26 15:30:00)
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        input_data = {
            "rev_request": rev_request,
            "msg_history": msg_history,
            "date_time": date_time,
        }
        chain_llm = prompt | model | StrOutputParser()
        answer_txt = chain_llm.invoke(input_data, config=config)
        answer = AIMessage(
            content=json.dumps(answer_txt, ensure_ascii=False), additional_kwargs={}
        )
        # Show logs
        msg = agent_msg_mgr.get_msg("ans_llm_solo", content=answer_txt)
        log.print(msg + "\n")
        return {"messages": [answer]}
