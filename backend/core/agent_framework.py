from typing import Dict, List, Optional, Any, TypedDict
from pathlib import Path
import operator
import time
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import BaseTool
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolExecutor
import re
import json
import logging

logger = logging.getLogger(__name__)

# Define state types
class AgentState(TypedDict):
    messages: List[Dict[str, str]]  # Chat history
    current_query: str              # Current user query
    retriever_result: Optional[str] # Result from retriever
    suggested_tool: Optional[str]   # Tool suggested for execution
    tool_input: Optional[Dict]      # Input parameters for tool
    requires_confirmation: bool     # Whether tool execution needs confirmation
    confirmed: bool                # Whether user confirmed tool execution
    final_response: Optional[str]   # Final response to return to user
    video_path: str               # Path to current video
    execution_history: List[Dict] # History of tool executions

class VideoAnalysisAgent:
    """Agent for handling video analysis queries and actions"""
    
    def __init__(self, llm, tools: List[Tool], retriever):
        """Initialize agent with components"""
        self.llm = llm
        self.tools = tools
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
        
        # Initialize tool executor
        self.tool_executor = ToolExecutor(tools)
        
        # Create the graph
        self.workflow = self._create_workflow()
        
        # Initialize RAG service and chain
        from backend.services.rag_service import RAGService
        self.rag_service = RAGService()
        self._current_chain = None
        
        # Define system prompt
        self.system_prompt = """You are a helpful AI assistant that analyzes videos and chats about them using RAG (Retrieval Augmented Generation).

Your primary role is to:
1. Use RAG to provide informed responses about video content
2. Suggest relevant analysis functions when they could provide more insights
3. Always get explicit confirmation before executing any actions
4. Maintain a natural, conversational tone while being precise

When responding:
- Default to using RAG to answer questions about video content
- If RAG data is insufficient, suggest running appropriate analysis
- For action requests, clearly explain what will happen and get confirmation
- Provide specific suggestions based on the conversation context

Available tools:
{tools}

Example interactions:
User: "What's happening in the video?"
Assistant: Let me check the analysis data... [provides RAG-based response]. I notice some interesting activity - would you like me to run a fresh scene analysis for more details?

User: "Are there any cars?"
Assistant: From the existing analysis, [RAG-based response about vehicles]. I could run object detection specifically focused on vehicles if you'd like more precise information. Would you like me to do that?

User: "Yes, run the detection"
Assistant: I'll run object detection to identify vehicles and other objects. This will analyze the current video frames. Would you like me to proceed with this analysis now?
"""

    def _create_workflow(self) -> StateGraph:
        """Create the agent workflow graph with checkpointing"""
        
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Define workflow nodes
        workflow.add_node("retrieve", self._retrieve_info)
        workflow.add_node("suggest_tool", self._suggest_tool)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("generate_response", self._generate_response)

        # Add edge from START to retrieve (entry point)
        workflow.add_edge(START, "suggest_tool")
        
        # Add conditional edges from retrieve
        workflow.add_conditional_edges(
            "suggest_tool",
            self._route_after_suggestion,
            {
                "retrieve": "retrieve",
                "execute": "execute_tool",
                "respond": "generate_response"
            }
        )
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "suggest": "suggest_tool",
                "respond": "generate_response"
            }
        )
        
        workflow.add_edge("execute_tool","generate_response")

        
        # Add edge to end
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("suggest_tool")
        
        return workflow
        

    def _retrieve_info(self, state: AgentState) -> AgentState:
        """Core retrieval logic"""
        query = state["current_query"]
        chat_history = state.get("messages", [])
        video_path = state.get("video_path")
        import pdb; pdb.set_trace()
        # Create/update knowledge base
        try:
            import pdb; pdb.set_trace()
            vectordb = self.rag_service.create_knowledge_base(Path(video_path))
            if not vectordb:
                return {
                    **state,
                    "retriever_result":"Found no analysis files. Do analysis on the video content.",
                    "error": "Failed to create/update knowledge base",
                    "messages": chat_history + [
                        {"role": "user", "content": query}
                    ]
                }
        
            # Query knowledge base
            result = self.rag_service.query_knowledge_base(
                query=query,
                chat_history=chat_history
            )
            
            retriever_response = result.get('result') if result else None
            
            # # Check if query suggests tool usage
            # tool_suggestion = self._analyze_for_tool_suggestion(query, retriever_response)
            
            return {
                **state,
                "retriever_result": retriever_response,
                # "suggested_tool": tool_suggestion["tool"] if tool_suggestion else None,
                # "tool_description": tool_suggestion["description"] if tool_suggestion else None,
                # "requires_confirmation": bool(tool_suggestion),
                "messages": chat_history + [
                    {"role": "user", "content": query}
                ]
            }
        except Exception as e:
            self.logger.error(f"Retrieval error: {e}")
            return {
                **state,
                "error": str(e),
                "messages": chat_history + [
                    {"role": "user", "content": query}
                ]
            }
        

    def _suggest_tool(self, state: AgentState) -> AgentState:
        """Analyze query and suggest appropriate tool"""
        # Create tool suggestion prompt with available tools
        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an interpreter for the user query for the video uploaded by him to do the following.
             1.Interpret the user query and context, and identify if the context is 
                a. to get information.
                b. to execute a tool or perform a specific analysis.
                c. approval or denial of a tool execution.
             2. If a retriever result is available, interpret and suggest an appropriate tool if needed.
                Available tools:
                {tools_description}
                Only suggest a tool if it would provide additional valuable information beyond what's in the retriever result.
                If suggesting a tool, explain why it would be helpful. if retriever result is not available and a tool should not be suggested, set the confirmed to False and requires_confirmation to True.
             3. If the user query appears to be a reply for tool execution question to confirm, and identifies user confirmation to the suggested tool,only then set the confirmed to True and requires_confirmation to False along with the tool. 
             4. If the user query requests and affirms for tool execution, and if the tool is available and capable of performing the user request, only then set the confirmed to True and requires_confirmation to False. Provide tool information if deemed eligible and available.

            You must respond with valid JSON in this exact format:
            {{{{"tool": "<tool_name or null if no tool needed>","confirmed": <boolean>,"requires_confirmation": <boolean>,"reason": "<explanation for suggesting or not suggesting a tool>"}}}}"""),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "Query: {query}\nretriever_result: {result}\n\nAnalyze the query and suggest a tool if appropriate. Return your response in the required JSON format.")
        ])
        #"input": {{{{"param": "value"}}}},
        import pdb; pdb.set_trace()
        # Get suggestion from LLM
        chain = prompt | self.llm | JsonOutputParser()
        response = chain.invoke({
            "messages": state["messages"],
            "query": state["current_query"],
            "result": state["retriever_result"]
        })
        # Extract and clean JSON from response
        try:
            # First try direct JSON parsing
            suggestion = json.loads(response.content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from markdown
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if json_match:
                try:
                    suggestion = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    suggestion = {
                        "tool": None,
                        "input": {},
                        "confirmed": False,
                        "requires_confirmation": True,
                        "reason": f"Failed to parse tool suggestion from JSON"
                    }
            else:
                # Fallback to default if no JSON found
                suggestion = {
                    "tool": None,
                    "input": {},
                    "confirmed": False,
                    "requires_confirmation": True,
                    "reason": "No valid JSON found in response"
                }
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            suggestion = {
                "tool": None,
                "input": {},
                "confirmed": False,
                "requires_confirmation": True,
                "reason": f"Failed to parse tool suggestion: {str(e)}"
            }

        # Validate suggested tool exists
        suggested_tool_name = suggestion.get("tool")
        if suggested_tool_name:
            matching_tool = next(
                (tool for tool in self.tools if tool.name == suggested_tool_name),
                None
            )
            if matching_tool:
                return {
                    **state,
                    "suggested_tool": suggested_tool_name,
                    "tool_input": suggestion.get("input"),
                    "tool_description": matching_tool.description,  # Add both for compatibility
                    "requires_confirmation": suggestion.get("requires_confirmation"),
                    "confirmed": suggestion.get("confirmed"),
                    "retriever_result": suggestion.get("reason")
                }
            
        # No valid tool suggested
        return {
            **state,
            "suggested_tool": None,
            "tool_input": None,
            "requires_confirmation": True,
            "confirmed": False
        }
        

    def _execute_tool(self, state: AgentState) -> AgentState:
        """Core tool execution logic"""
        if not state["confirmed"]:
            tool_desc = state.get("tool_description", "this action")
            return {
                **state,
                "final_response": f"Would you like me to {tool_desc}? Please confirm.",
                "requires_confirmation": True
            }
            
        tool_name = state["suggested_tool"]
        
        # Find matching tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return {
                **state,
                "error": f"Tool {tool_name} not found",
                "final_response": f"Sorry, the tool {tool_name} is not available.",
                "requires_confirmation": False  # Reset confirmation since tool not found
            }
            
        try:
            # Execute tool with current state context
            video_path = state.get("video_path")
            result = tool._run(Path(video_path))
            
            return {
                **state,
                "retrieved_result": result,
                "messages": state["messages"] + [
                    {"role": "system", "content": f"Executed {tool_name}: {result}"}
                ]
            }
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return {
                **state,
                "error": str(e),
                "final_response": f"Error executing {tool_name}: {str(e)}"
            }


    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response combining retriever and tool results"""
        import pdb; pdb.set_trace()
        if state.get("error"):
            return {
                **state,
                "final_response": f"I encountered an error: {state['error']}"
            }
        
        if state.get("suggested_tool") and not state.get("confirmed"):
            tool_desc =[tool.description for tool in self.tools if tool.name == state["suggested_tool"]][0].lower()
            # tool_desc = state.get("tool_desc", "this action")
            final_response = state["retriever_result"]+". "+f"Would you like me to {tool_desc}? Please confirm."
            return {
                **state,
                "final_response": final_response,
                "messages": state["messages"] + [
                {"role": "assistant", "content": f"Would you like me to {tool_desc}? Please confirm."}]
            }
            
        # Create response prompt based on state
        prompt_parts = [
            "Based on the following information, provide a helpful response. If a tool has been executed, Say that the analysis is completed. Summarize the tool results and request to ask any questions.",
            f"Query: {state['current_query']}",
            f"Retrieved Context: {state.get('retriever_result', 'No context available')}"
        ]
            
        # if state.get("tool_result"):
        #     prompt_parts.append(f"Say that the analysis is completed. Summarize the tool results and request to ask any questions. Tool Result: {state['tool_result']}") 


        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a helpful response using all available information."),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "\n".join(prompt_parts))
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "messages": state["messages"],
            "query": state["current_query"]
        })
        
        # Extract content from AIMessage or string
        response_text = response.content if hasattr(response, 'content') else json.jsonify(response)
        
        return {
            **state,
            "final_response": response_text,
            "messages": state["messages"] + [
                {"role": "assistant", "content": response_text}
            ]
        }
            
    def _analyze_for_tool_suggestion(self, query: str, retriever_response: Optional[str]) -> Optional[Dict]:
        """Analyze if query suggests using a specific tool"""
        tool_patterns = {
            'scene_analysis': {
                'patterns': ['analyze scene', 'what is happening', 'describe scene', 'what do you see', 'tell me about the video', 'what is in the video', 'search the video', 'explain the video'],
                'description': 'Perform detailed scene analysis'
            },
            'object_detection': {
                'patterns': ['find objects', 'detect objects', 'locate', 'identify objects'],
                'description': 'Detect and locate objects in the scene'
            },
            'edge_detection': {
                'patterns': ['show edges', 'detect edges', 'highlight boundaries', 'outline'],
                'description': 'Highlight edges and boundaries in the scene'
            }
        }
        
        query_lower = query.lower()
        
        for tool_name, config in tool_patterns.items():
            if any(pattern in query_lower for pattern in config['patterns']):
                return {
                    'tool': tool_name,
                    'description': config['description']
                }
                
        return None

    def _route_after_retrieve(self, state: AgentState) -> str:
        """Route to next node after retrieval"""
        if state.get("error"):
            if state.get("error") == "Failed to create/update knowledge base" and state.get("retriever_result"):
                return "suggest"
            return "respond"
        if not state.get("error") and not state.get("confirmed"):
            return "suggest"
        return "respond"

    def _route_after_suggestion(self, state: AgentState) -> str:
        """Route to next node after tool suggestion"""
        if state.get("error") or ( not state.get("confirmed") and state.get("suggested_tool")):
            return "respond"
        if not state["requires_confirmation"] and state["confirmed"]:
            return "execute" 
        return "retrieve"
    
        
    """Agent for handling video analysis queries and actions
    
    The workflow can be invoked directly using:
    ```python
    state = AgentState(
        messages=chat_history or [],
        current_query=query,
        retriever_result=None,
        suggested_tool=None,
        tool_input=None,
        requires_confirmation=True,
        confirmed=False,
        final_response=None,
        video_path=video_path,
        execution_history=[]
    )
    result = workflow.invoke(state)
    ```
    """
