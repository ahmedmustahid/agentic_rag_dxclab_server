import datetime
import json
import os
import time
from pathlib import Path

import arxiv
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.config import get_stream_writer
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, StreamWriter
from typing_extensions import Annotated

from src.routers.agentic_rag.message_utils import MsgUtils
from src.routers.agentic_rag.param_llm import get_gpt_model
from src.routers.agentic_rag.search_answer import SearchAnswerEngine
from src.routers.agentic_rag.state import State
from src.routers.utils.agent_msg_manager import AgentMsgManager
from src.routers.utils.log_dev import LogDev
from src.routers.utils.prompt_manager import PromptManager

log = LogDev()
load_dotenv()

prompt_mgr = PromptManager()
agent_msg_mgr = AgentMsgManager()
msg_util = MsgUtils()

AGENT_THOUGHT_LANG = os.environ.get("AGENT_THOUGHT_LANG", "en")
RAG_INDEX_LANG = os.environ.get("RAG_INDEX_LANG", "en")

# Max plan
max_plan = int(os.environ.get("MAX_PLAN", 5))
# Max turn
max_turn = int(os.environ.get("MAX_TURN", 2))
# Max search text
max_search_txt = int(os.environ.get("MAX_SEARCH_TXT", 10000))

# get_stream_writer manual: [How to stream data from within a tool](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/)
model = get_gpt_model()
# To clearly communicate the information contained in the vector database and to prevent it from being used for other purposes
vector_db_info = "Internal company information for the fictional companies Fic-GreenLife, Fic-NextFood, and Fic-TechFrontier"

# Get vector store
HUG_EMBE_MODEL_NAME = os.getenv("HUG_EMBE_MODEL_NAME")
embeddings = HuggingFaceEmbeddings(model_name=HUG_EMBE_MODEL_NAME)
if RAG_INDEX_LANG.lower() == "ja":
    index_path = Path("src/routers/agentic_rag/index/ja").resolve()
else:
    index_path = Path("src/routers/agentic_rag/index/en").resolve()
vector_store = FAISS.load_local(
    index_path, embeddings, allow_dangerous_deserialization=True
)

# API key of Tavily search
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = os.getenv("TAVILY_MAX_RESULTS")
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH")

# Write tool information. LLM read it and select tools to call.
tool_info = """
## Search Tools
The following tools are capable of searching only. Please do not call them for any other purpose.

- **ans_tavily**: Can perform web searches.
- **ans_arxiv**: Can search for papers published on arXiv.
- **search_rag**: Can only search the internal information of the three companies Fic-GreenLife, Fic-NextFood, and Fic-TechFrontier. It does not hold any information on other companies. If you need information on other companies, please use a different tool.

## Tools for LLM-Generated Responses
- **ans_llm_base**: Provides answers based on the LLM's own knowledge. It is capable of answering questions, summarizing, taking action on research findings, analyzing, extracting keywords, computing, offering opinions and insights, reasoning, and listing results.
"""


class SearchError(Exception):
    """Exception thrown when an unknown error occurs during search processing"""

    pass


class AutoResearchAgent:
    """
    AutoResearchAgent
    Create plans and conduct investigations autonomously
    """

    def __init__(self):
        pass

    def invoke_with_retry(self, chain, input_data, config, max_retries=3, sleep_time=1):
        """
        This function executes chain.invoke and retries if an OutputParserException occurs.

        Parameters:
          chain: Chain to be executed
          input_data: Input data to be passed to chain.invoke
          config: Settings to be passed to chain.invoke
          max_retries (int): Maximum number of retries (default is 3)
          sleep_time (int or float): Wait time before retrying (seconds, default is 1 second)

        Returns:
          plan_json: Result of successful execution of chain.invoke

        Raises:
          Exception: Raised when the maximum number of retries is exceeded
        """
        for attempt in range(max_retries):
            try:
                plan_json = chain.invoke(input_data, config=config)
                return plan_json  # If successful, return the result
            except OutputParserException as e:
                print(
                    f"OutputParserException occurred: {e}. Retrying ({attempt+1}/{max_retries})"
                )
                time.sleep(sleep_time)
        raise Exception("Error: LLM call retry limit reached.")

    @staticmethod
    @tool
    def ans_llm_base(
        state: Annotated[State, InjectedState], config: RunnableConfig
    ) -> str:
        """
        Answer using LLM's own knowledge. To provide direct answers based on general knowledge and FAQs.
        I can answer questions, take actions based on research results, analyze, extract keywords, perform calculations, provide opinions, think, list results, and summarize search results using the knowledge I possess as a large language model.

        Tool Specifications:
          Example Use:
          - The user asks questions like "What does XYZ mean?" or other queries based on general knowledge.
          - Extracting keywords from the document.
          - Summarizing the results of the research result so far and extract the important points.

        Args:
          state: Annotated[State, InjectedState]:

        Returns:
          str: answer
        """
        log.print("- Start: ans_llm_base")
        messages = state["messages"]
        # Get survey results (starting from type:start_turn)
        res_history = msg_util.get_history(messages)
        # Prompt
        prompt_template = prompt_mgr.get_prompt("ans_llm_base")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "res_history", "date_time"],
        )
        question = state["plan_exec"]["plan_exec"]
        # Get the current date and time
        now = datetime.datetime.now()
        # Specify the date and time format (e.g. 2025-03-26 15:30:00)
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        input_data = {
            "question": question,
            "res_history": res_history,
            "date_time": date_time,
        }
        chain_llm = prompt | model | StrOutputParser()
        answer = chain_llm.invoke(input_data, config=config)
        return answer

    @staticmethod
    @tool
    def search_rag(
        state: Annotated[State, InjectedState], config: RunnableConfig
    ) -> str:
        """
        Answer only internal information of 3 fictitious companies named Fic-GreenLife, Fic-NextFood and Fic-TechFrontier. When you want other companies information, use other tools because It can not answer other companies.

        Tool Specifications:
          - Purpose: When the user asks questions only for 3 companies like "What isFic-GreenLife?".

        Args:
          state: Annotated[State, InjectedState]
          config: RunnableConfig

        Returns:
          str: answer
        """
        log.print("\n<<Start: search_rag>>")
        question = state["plan_exec"]["plan_exec"]
        top_k = 3

        try:
            # When using Hugging face embedding
            results = vector_store.similarity_search_with_score(
                question, k=top_k, filter=None
            )
            doc_cnt = 1
            answer = ""
            for result in results:
                title = result[0].metadata["title"]
                content = result[0].page_content
                answer += f"## Title: {title}\n{content}\n\n"
                doc_cnt += 1
        except Exception as err:
            err_str = f"Error: A problem occurred while searching. {err}"
            raise SearchError(err_str) from err

        # Show logs
        msg = agent_msg_mgr.get_msg("search_rag", content=answer)
        log.print(msg + "\n")

        return answer

    @staticmethod
    @tool
    def ans_tavily(
        state: Annotated[State, InjectedState], config: RunnableConfig
    ) -> str:
        """
        Answer using tavily web search. Retrieve documents from web sites.

        Tool Specifications:
          - Purpose: To search and retrieve relevant information from web sites in response to user queries.

        Args:
          state: Annotated[State, InjectedState]
          config: RunnableConfig

        Returns:
          str: answer
        """
        try:
            top_k = TAVILY_MAX_RESULTS
            log.print("\n<<Start: ans_tavily>>")
            question = state["plan_exec"]["plan_exec"]
            if question == None:
                raise ValueError("Error: There are no questions for tavily search.")
            sa_ins = SearchAnswerEngine()
            retriever = TavilySearchAPIRetriever(
                k=top_k,
                search_depth=TAVILY_SEARCH_DEPTH,
                include_answer=True,
                include_raw_content=False,
            )
            results = retriever.invoke(question)
            doc_cnt = 1
            answer = ""
            for result in results:
                title = sa_ins.repair_enc_univ(result.metadata["title"])
                content = sa_ins.repair_enc_univ(
                    sa_ins.truncate_text(result.page_content, max_search_txt)
                )
                url = result.metadata["source"]
                answer += (
                    f"## Title: {title}\n### URL:{url}\n### Content:\n{content}\n\n"
                )
                doc_cnt += 1
        except Exception as err:
            err_str = f"Error: A problem occurred while searching. {err}"
            raise SearchError(err_str) from err

        # Show logs
        msg = agent_msg_mgr.get_msg("ans_tavily", content=answer)
        log.print(msg + "\n")
        return answer

    @staticmethod
    @tool
    def ans_arxiv(
        state: Annotated[State, InjectedState], config: RunnableConfig
    ) -> str:
        """
        Use only to search arxiv.
        Answer using arxiv search. Retrieve documents from arxiv. Use this tool only to search for physics, mathematics, computer matters.

        arXiv is an open-access repository for scholarly articles.
        arXiv currently serves the fields of physics, mathematics, computer science.

        Tool Specifications:
          - Purpose: To search and retrieve relevant information of scholarly articles.

        Args:
          state: Annotated[State, InjectedState]
          config: RunnableConfig

        Returns:
          str: answer
        """
        log.print("\n<<Start: ans_arxiv>>")
        # Get survey results (starting from type:start_turn)
        messages = state["messages"]
        res_history = msg_util.get_history(messages)
        writer = get_stream_writer()
        question = state["plan_exec"]["plan_exec"]
        # Create query of arxiv
        # Prompt
        prompt_template = prompt_mgr.get_prompt("ans_arxiv")
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["question", "res_history"]
        )
        input_data = {"question": question, "res_history": res_history}
        chain_llm = prompt | model | StrOutputParser()
        query = chain_llm.invoke(input_data, config=config)
        writer(f"[arxiv query]\n{query}")
        try:
            search = arxiv.Search(
                query=query, max_results=5, sort_by=arxiv.SortCriterion.SubmittedDate
            )
            # Create a Client object and pass in the Search object
            client = arxiv.Client()
            results = client.results(search)

            doc_cnt = 1
            answer = ""
            for result in results:
                title = result.title
                published = result.published
                summary = result.summary
                url = result.entry_id
                answer += f"## Title: {title}, Published: {published}, Url: {url} \n{summary}\n\n"
                doc_cnt += 1
        except Exception as err:
            err_str = f"Error: A problem occurred while searching. {err}"
            raise SearchError(err_str) from err

        # Show logs
        msg = agent_msg_mgr.get_msg("ans_arxiv", content=answer)
        log.print(msg + "\n")
        return answer

    # --- Agent functions ---

    def create_plan(
        self,
        state: Annotated[State, InjectedState],
        config: RunnableConfig,
        writer: StreamWriter,
    ) -> dict:
        """
        Develop a work plan to achieve the objectives of users in response to their questions and requests

        Args:
          state: Annotated[State, InjectedState]
          config: RunnableConfig

        Returns:
          dict: messages
        """
        log.print("\n<<Start: create_plan>>")
        messages = state["messages"]
        # Record the initial point of the conversation (as a starting point for creating a conversation history)
        plan_status_json = {"type": "start_turn"}
        start_turn = AIMessage(
            content=json.dumps(plan_status_json, ensure_ascii=False),
            additional_kwargs={},
        )
        turn = 1
        # Prompt
        prompt_template = prompt_mgr.get_prompt("create_plan")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "max_plan",
                "rev_request",
                "msg_history",
                "vector_db_info",
                "date_time",
                "tool_info",
            ],
        )
        msg_history = msg_util.get_pure_msg(messages)
        rev_request = state["rev_request"]
        # Get the current date and time
        now = datetime.datetime.now()
        # Specify the date and time format (e.g. 2025-03-26 15:30:00)
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        input_data = {
            "max_plan": max_plan,
            "rev_request": rev_request,
            "msg_history": msg_history,
            "vector_db_info": vector_db_info,
            "date_time": date_time,
            "tool_info": tool_info,
        }
        chain = prompt | model | JsonOutputParser()
        plan_json = self.invoke_with_retry(chain, input_data, config=config)
        # Get plan_status
        plan_status = plan_json.get("plan_status")
        num_plans = len(plan_json.get("plan"))
        # If the number of plans exceeds max_plan, force a single plan and set rev_request
        # because Too many plans will waste resources
        if num_plans > max_plan:
            plan_status = ["open"]
            plan_json["plan_status"] = plan_status
            plan_json["plan"] = [rev_request]
            plan_over = True
            num_plans = 1
        else:
            plan_over = False
        # Show logs
        plans_txt = ""
        for i, plan in enumerate(plan_json.get("plan"), start=1):
            plans_txt += f" - Plan{i}: {plan}"
            if i != num_plans:
                plans_txt += "\n"
        msg = agent_msg_mgr.get_msg("create_plan", num_plans=num_plans, plans=plans_txt)
        log.print(msg + "\n")
        # Streaming custom message
        writer(msg)
        return {
            "messages": [start_turn],
            "turn": turn,
            "plan_status": plan_status,
            "plan": plan_json,
            "plan_over": plan_over,
        }

    def update_plan_status(self, state: Annotated[State, InjectedState]) -> dict:
        """
        Update the plan status, changing the first occurrence of open to done.
        """
        log.print("\n<<Start: update_plan_status>>")
        # Detects errors caused by tool calling and returns an exception if an error occurs.
        messages = state["messages"]
        latest_tool_msg = msg_util.get_latest_ai_tool_msg(messages)
        if latest_tool_msg.startswith("Error: SearchError"):
            raise SearchError(latest_tool_msg)
        plan_status = state["plan_status"]
        # Update status
        # Since the plan is executed in order from the top, the first open value found is changed to done.
        for index, status in enumerate(plan_status):
            if status == "open":
                plan_status[index] = "done"
                break
        # Show logs
        num_plans = len(plan_status)
        plan_st_txt = ""
        for i, status in enumerate(plan_status, start=1):
            if status == "done":
                status_show = "Done"
            else:
                status_show = "Open"
            plan_st_txt += f"- Plan{i}: {status_show}"
            if i != num_plans:
                plan_st_txt += "\n"
        msg = agent_msg_mgr.get_msg("update_plan_status", plan_status=plan_st_txt)
        log.print(msg + "\n")
        return {"plan_status": plan_status}

    def check_open_plan(self, state: Annotated[State, InjectedState]) -> bool:
        """
        Check if a plan with status open exists
        """
        log.print("- Start: check_open_plan")
        if state["plan_status"] is None:
            msg = "Error: plan_status does not exist."
            answer = AIMessage(
                content=json.dumps(msg, ensure_ascii=False), additional_kwargs={}
            )
            return {"messages": [answer]}
        # Process to obtain plan based on index information
        # Get the first index where plan_status is 'open'
        index = next(
            (i for i, status in enumerate(state["plan_status"]) if status == "open"),
            None,
        )
        plan_js = state["plan"]
        if index is not None:
            if len(plan_js["plan"]) < index:
                # When the number of statuses is greater than the number of plans, resulting in an index that exceeds the number of plans.
                index = None  # Assume all plans are complete
        if index == None:
            result = False
            log.print("There is no open plan.")
        else:
            result = True
            log.print("There is an open plan.")
        return result

    def create_final_answer(
        self, state: Annotated[State, InjectedState], config: RunnableConfig
    ) -> str:
        """
        Creating a final answer

        Args:
          state: Annotated[State, InjectedState]
          config: RunnableConfig

        Returns:
          dict: messages
        """
        log.print("\n<<Start: create_final_answer>>")
        messages = state["messages"]
        # Get survey results (starting from type:start_turn)
        res_history = msg_util.get_history(messages)
        # Prompt
        prompt_template = prompt_mgr.get_prompt("create_final_answer")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["request", "res_history", "msg_history", "date_time"],
        )
        msg_history = msg_util.get_pure_msg(messages)
        # Get the current date and time
        now = datetime.datetime.now()
        # Specify the date and time format (e.g. 2025-03-26 15:30:00)
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        # Latest Questions and Requests from Users (Updated)
        rev_request = state["rev_request"]
        input_data = {
            "rev_request": rev_request,
            "res_history": res_history,
            "msg_history": msg_history,
            "date_time": date_time,
        }
        chain = prompt | model | StrOutputParser()
        answer_llm = chain.invoke(input_data, config=config)
        answer = AIMessage(
            content=json.dumps(answer_llm, ensure_ascii=False), additional_kwargs={}
        )
        return {"messages": [answer]}

    def judge_replan(
        self,
        state: Annotated[State, InjectedState],
        config: RunnableConfig,
        writer: StreamWriter,
    ) -> Command:
        """
        Decide whether to replan

        Returns:
          boolean: True: Re-plan, False: Do not re-plan
        """
        log.print("<<Start: judge_replan>>")
        messages = state["messages"]
        turn = state["turn"]
        if turn == 0 or turn == "":
            answer = "Error: The number of turns is 0."
            print(answer)
            raise ValueError(answer)
        judge = True
        if turn >= max_turn:
            # Do not re-plan (the limit has been exceeded)
            judge = False
        else:
            # --- Determine if the survey results contain the answer ---
            # Get survey results (starting from type:start_turn)
            res_history = msg_util.get_history(messages)
            # Prompt
            prompt_template = prompt_mgr.get_prompt("judge_replan")
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["request", "res_history", "msg_history", "date_time"],
            )
            # Latest Questions and Requests from Users (Updated)
            rev_request = state["rev_request"]
            msg_history = msg_util.get_pure_msg(messages)
            # Get the current date and time
            now = datetime.datetime.now()
            # Specify the date and time format (e.g. 2025-03-26 15:30:00)
            date_time = now.strftime("%Y-%m-%d %H:%M:%S")
            input_data = {
                "rev_request": rev_request,
                "res_history": res_history,
                "msg_history": msg_history,
                "date_time": date_time,
            }
            chain = prompt | model | JsonOutputParser()
            answer_json = chain.invoke(input_data, config=config)
            value = answer_json.get("is_included")
            if value == "yes":
                # Do not replan if answer is included
                judge = False
            else:
                # Otherwise, replan.
                judge = True
        goto = ""
        if judge == True:
            # Replan
            goto = "create_revised_plan"
            if AGENT_THOUGHT_LANG.lower() == "ja":
                judge_msg = f"判定: 再プランを行います。回答が見つからないため。"
            else:
                judge_msg = f"Verdict: Perform replanning because no answer was found."
        else:
            # Final answer
            goto = "create_final_answer"
            if AGENT_THOUGHT_LANG.lower() == "ja":
                judge_msg = f"判定: 再プランを行いません。"
            else:
                judge_msg = f"Verdict: Do not perform replanning."

        msg = agent_msg_mgr.get_msg("judge_replan", turn=turn, content=judge_msg)
        # Show logs
        log.print(msg + "\n")
        # Streaming custom message
        writer(msg)
        return Command(goto=goto)

    def create_revised_plan(
        self,
        state: Annotated[State, InjectedState],
        config: RunnableConfig,
        writer: StreamWriter,
    ) -> dict:
        """
        Create an updated plan

        Args:
          state: Annotated[State, InjectedState]
          config: RunnableConfig

        Returns:
          dict: messages
        """

        log.print(" << Start: create_revised_plan >>")
        # Record the initial point of the conversation (as a starting point for creating a conversation history)
        plan_status_json = {"type": "start_turn"}
        start_turn = AIMessage(
            content=json.dumps(plan_status_json, ensure_ascii=False),
            additional_kwargs={},
        )

        turn = state["turn"]
        if turn == 0 or turn == "":
            answer = "Error: The number of turns is 0."
            print(answer)
            raise ValueError(answer)
        turn += 1
        # Prompt
        prompt_template = prompt_mgr.get_prompt("create_revised_plan")
        plan = state["plan"]
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "max_plan",
                "vector_db_info",
                "plan",
                "date_time",
                "tool_info",
                "rev_request",
            ],
        )
        rev_request = state["rev_request"]
        # Get the current date and time
        now = datetime.datetime.now()
        # Specify the date and time format (e.g. 2025-03-26 15:30:00)
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        input_data = {
            "max_plan": max_plan,
            "vector_db_info": vector_db_info,
            "plan": plan,
            "date_time": date_time,
            "tool_info": tool_info,
            "rev_request": rev_request,
        }
        chain_llm = prompt | model | JsonOutputParser()
        replan_json = chain_llm.invoke(input_data, config=config)
        # Get plan_status
        plan_status = replan_json.get("plan_status")

        num_plans = len(replan_json.get("plan"))

        # If the number of plans exceeds max_plan, force a single plan and set rev_request.
        # Too many plans will waste resources
        if num_plans > max_plan:
            plan_status = ["open"]
            replan_json["plan_status"] = plan_status
            replan_json["plan"] = [rev_request]
            plan_over = True
            num_plans = 1
        else:
            plan_over = False
        # Show logs
        plans_txt = ""
        for i, plan in enumerate(replan_json.get("plan"), start=1):
            plans_txt += f" - Plan{i}: {plan}"
            if i != num_plans:
                plans_txt += "\n"
        msg = agent_msg_mgr.get_msg(
            "create_revised_plan", num_plans=num_plans, plans=plans_txt
        )
        log.print(msg + "\n")
        # Streaming custom message
        writer(msg)
        answer = AIMessage(
            content=json.dumps(replan_json, ensure_ascii=False), additional_kwargs={}
        )
        return {
            "messages": [start_turn, answer],
            "turn": turn,
            "plan": replan_json,
            "plan_status": plan_status,
            "plan_over": plan_over,
        }
