"""All Pydantic models used across the system."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """Represents a tool invocation decision made by an agent."""

    tool: Literal["rag_search", "calculator", "unit_converter", "web_search", "final_answer"]
    args: dict
    reasoning: str  # why the agent chose this tool


class RAGResult(BaseModel):
    """Result returned by the RAG Specialist agent."""

    answer: str
    source_chunks: list[str]
    source_files: list[str]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


class WebSearchResult(BaseModel):
    """Result returned by the web search tool."""

    query: str
    results: list[dict]  # each dict: {title, url, snippet}
    summary: str          # LLM-synthesised summary of the search results
    confidence: Literal["high", "medium", "low"]


class ResearchPlan(BaseModel):
    """Query decomposition produced by the research planner."""

    is_conversational: bool = False  # greetings, chit-chat, meta questions
    sub_questions: list[str] = []
    tool_call: ToolCall | None = None  # set when a specific tool should be invoked directly


class GapAnalysis(BaseModel):
    """Gap analysis produced after reviewing all gathered evidence."""

    model_config = ConfigDict(populate_by_name=True)

    is_sufficient: bool = Field(alias="isSufficient", default=False)
    reasoning: str = ""
    follow_up_questions: list[str] = Field(alias="gaps", default=[])  # additional doc-retrieval questions
    web_search_queries: list[str] = Field(alias="webSearchQueries", default=[])   # questions needing live web data


class OrchestratorDecision(BaseModel):
    """A single routing decision made by the Orchestrator."""

    action: Literal[
        "use_calculator",
        "use_web_search",
        "answer_directly",
        "done",
    ]
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    subtask: str | None = None  # populated when delegating


class AgentStep(BaseModel):
    """One step in the agent reasoning trace."""

    agent: str
    action: str
    input: str
    output: str
    tool_used: str | None = None
    thinking: str = ""  # raw <think> content from the LLM, if any


class FinalResponse(BaseModel):
    """The complete response returned to the caller."""

    answer: str
    steps: list[AgentStep]    # full reasoning trace
    sources: list[str]         # document source files used
    web_sources: list[str]     # URLs returned by web search (empty if unused)
    confidence: str
    from_memory: bool          # whether the answer came from long-term memory
    contexts: list[str] = []   # raw retrieved chunk texts (used by eval pipeline)


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    question: str
    user_id: str = Field(
        ...,
        description="Opaque client-generated UUID identifying the user.",
        min_length=1,
    )
