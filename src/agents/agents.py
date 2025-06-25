from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from typing import Union, List


class Agents:
    """
    The base class to create an agent.
    """
    @classmethod
    def get(cls,
            llm: Union[ChatOpenAI, HuggingFaceEndpoint],
            tools: List[Tool],
            prompt: PromptTemplate,
            react: bool,
            verbose: bool = False) -> AgentExecutor:

        if react:
            print("[INFO] Creating React Agent.")
            agent = create_react_agent(llm, tools, prompt)
            return AgentExecutor(
                agent=agent,
                tools=tools,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                verbose=verbose
            )
        else:
            raise NotImplementedError(
                "Other prompt style implementation is needed."
            )  
