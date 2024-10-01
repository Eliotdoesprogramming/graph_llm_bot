import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from agent.tools import tools
from agent.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agent.research_assistant import researcher_instructions
from langchain_core.runnables import RunnableLambda

class AgentState(MessagesState):
    safety: LlamaGuardOutput
    is_last_step: IsLastStep


# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True),
}

if os.getenv("GROQ_API_KEY") is not None:
    models["llama-3.1-70b"] = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
if os.getenv("GOOGLE_API_KEY") is not None:
    models["gemini-1.5-flash"] = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.5, streaming=True
    )





def wrap_model(model: BaseChatModel, instructions: str = "") -> RunnableLambda:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig):
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m, instructions=researcher_instructions)
    response = await model_runnable.ainvoke(state, config)



    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

async def llama_guard_output(state: AgentState, config: RunnableConfig):
    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}
    else:
        return {"messages":[], "safety": safety_output}
async def llama_guard_input(state: AgentState, config: RunnableConfig):
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig):
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}

async def confirm_tools(state:AgentState, config:RunnableConfig):
    tool_calls = state["messages"][-1].tool_calls
    return {"messages": [AIMessage(content=f"I would like to use the following tool: {tool_calls[0]['name']}.\n Is that okay? (yes/no)")]}


async def continue_to_tool(state:AgentState, config:RunnableConfig):
    """If the user confirms the tool, re-add the tool call to the messages."""
    tool_call_message = state["messages"][-1]
    return {"messages": [tool_call_message]}


# Define the graph
builder = StateGraph(AgentState)
builder.add_node("guard_input", llama_guard_input)
builder.set_entry_point("guard_input")
builder.add_node("block_unsafe_content", block_unsafe_content)
builder.add_node("model", acall_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("continue_to_tool", continue_to_tool)
builder.add_node("llama_guard_output", llama_guard_output)
builder.add_node("confirm_tools", confirm_tools)

# Check for unsafe input and block further processing if found
def check_safety(state: AgentState):
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
    
    if type(state["messages"][-1])==AIMessage and state["messages"][-1].tool_calls:
        return "tools"
    return "safe"

def check_tool_approval(state: AgentState):
    if type(state["messages"][-1])==AIMessage and state["messages"][-1].content.lower() == "yes":
        return "approved"
    return "denied"

builder.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)
builder.add_edge("model", "llama_guard_output")
builder.add_conditional_edges(
    "llama_guard_output", check_safety, {"unsafe": "block_unsafe_content", "safe": END, "tools": "confirm_tools"}
)
builder.add_conditional_edges(
    "confirm_tools", check_tool_approval, {"approved": "continue_to_tool", "denied": END})

builder.add_edge("continue_to_tool", "tools")
# Always END after blocking unsafe content
builder.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
builder.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return "done"


#builder.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=['confirm_tools'],
    debug=True
)
with open("agent_diagram.png", "wb") as f:
    f.write(graph.get_graph(xray=True).draw_mermaid_png())

if __name__ == "__main__":
    import asyncio
    from uuid import uuid4
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        inputs = {"messages": [("user", "Find me a recipe for chocolate chip cookies")]}
        result = await graph.ainvoke(
            inputs,
            config=RunnableConfig(configurable={"thread_id": uuid4()}),
        )
        result["messages"][-1].pretty_print()

        # Draw the agent graph as png
        # requires:
        # brew install graphviz
        # export CFLAGS="-I $(brew --prefix graphviz)/include"
        # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
        # pip install pygraphviz
        #
        # research_assistant.get_graph().draw_png("agent_diagram.png")

    asyncio.run(main())
