import datetime

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, StreamWriter
from typing_extensions import Annotated

from src.routers.agentic_rag.message_utils import MsgUtils
from src.routers.agentic_rag.param_llm import get_gpt_model
from src.routers.agentic_rag.state import State
from src.routers.utils.agent_msg_manager import AgentMsgManager
from src.routers.utils.log_dev import LogDev
from src.routers.utils.prompt_manager import PromptManager

agent_msg_mgr = AgentMsgManager()

log = LogDev()
load_dotenv()

model = get_gpt_model()
prompt_mgr = PromptManager()
msg_util = MsgUtils()


class RouterAgent:
    """
    RouterAgent
    Decide which Agent to call in response to user questions and requests
    """

    def __init__(self):
        pass

    def route_request(
        self,
        state: Annotated[State, InjectedState],
        config: RunnableConfig,
        writer: StreamWriter,
    ) -> Command:
        """
        Decide which Agent to call in response to user questions and requests

        Args:
          state: State
          config: RunnableConfig

        Returns:
          dict: messages
        """
        log.print("\n<<Start: route_query>>")
        messages = state["messages"]
        request = msg_util.get_latest_human_msg(messages)
        # Prompt
        prompt_template = prompt_mgr.get_prompt("route_request")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["request", "msg_history", "date_time"],
        )
        msg_history = msg_util.get_pure_msg(messages)
        # Get the current date and time
        now = datetime.datetime.now()
        # Specify the date and time format (e.g. 2025-03-26 15:30:00)
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        input_data = {
            "request": request,
            "msg_history": msg_history,
            "date_time": date_time,
        }
        chain = prompt | model | JsonOutputParser()
        router_json = chain.invoke(input_data, config=config)
        rev_request = router_json["revised_request"]
        # Show logs
        msg = agent_msg_mgr.get_msg(
            "route_request",
            request=request,
            rev_request=rev_request,
            agent_name=router_json["agent_name"],
            reason_sel=router_json["reason_sel"],
            revised_reason=router_json["revised_reason"],
        )
        log.print(msg + "\n")
        # Streaming custom message
        writer(msg)
        if router_json["agent_name"] == "answer_llm":
            goto = "ans_llm_solo"
        elif router_json["agent_name"] == "auto_research":
            goto = "create_plan"
        elif router_json["agent_name"] == "ask_human":
            goto = "ask_human"
        else:
            msg = "Error: No AI Agent to execute."
            print(msg)
            raise ValueError(msg)
        return Command(update={"rev_request": rev_request}, goto=goto)
