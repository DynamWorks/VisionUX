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
from langgraph.graph import StateGraph, END
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
    state_id: str                 # Unique state identifier
    last_checkpoint: float        # Last checkpoint timestamp
    execution_history: List[Dict] # History of tool executions

class VideoAnalysisAgent:
    """Agent for handling video analysis queries and actions"""
    
    def __init__(self, llm, tools: List[Tool], retriever):
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
        
        # Define nodes with checkpointing
        workflow.add_node("retrieve", self._retrieve_info_with_checkpoint)
        workflow.add_node("suggest_tool", self._suggest_tool_with_checkpoint)
        workflow.add_node("execute_tool", self._execute_tool_with_checkpoint)
        workflow.add_node("generate_response", self._generate_response_with_checkpoint)
        
        # Define edges
        workflow.add_edge("retrieve", "suggest_tool")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "suggest_tool",
            self._route_after_suggestion,
            {
                "execute": "execute_tool",
                "respond": "generate_response"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_tool",
            self._route_after_execution,
            {
                "respond": "generate_response",
                "suggest": "suggest_tool"
            }
        )
        
        # Add edge to end
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        return workflow
            
    def _retrieve_info_with_checkpoint(self, state: AgentState) -> AgentState:
        """Use retriever to get initial response with checkpointing"""
        try:
            # Load checkpoint if exists
            checkpointer = state.get('config', {}).get('checkpointer')
            if checkpointer:
                checkpoint = checkpointer.get_checkpoint(f"retrieve_{state['state_id']}")
                if checkpoint:
                    return checkpoint

            # Execute retrieval
            result = self._retrieve_info(state)

            # Save checkpoint
            if checkpointer:
                checkpointer.save_checkpoint(f"retrieve_{state['state_id']}", result)
                result['last_checkpoint'] = time.time()

            return result
        except Exception as e:
            self.logger.error(f"Retrieval error: {e}")
            return {**state, 'error': str(e)}

    def _retrieve_info(self, state: AgentState) -> AgentState:
        """Core retrieval logic"""
        query = state["current_query"]
        chat_history = state.get("messages", [])
        
        # Get retriever response
        try:
            result = self.retriever.get_relevant_documents(query)
            retriever_response = result[0].page_content if result else None
            
            # Check if query suggests tool usage
            tool_suggestion = self._analyze_for_tool_suggestion(query, retriever_response)
            
            return {
                **state,
                "retriever_result": retriever_response,
                "suggested_tool": tool_suggestion["tool"] if tool_suggestion else None,
                "tool_description": tool_suggestion["description"] if tool_suggestion else None,
                "requires_confirmation": bool(tool_suggestion),
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
        # Create tool suggestion prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the query and retriever result to suggest an appropriate tool if needed."),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "Query: {query}\nRetriever result: {result}\n\nShould I suggest a tool?")
        ])
        
        # Get suggestion from LLM
        chain = prompt | self.llm | JsonOutputParser()
        suggestion = chain.invoke({
            "messages": state["messages"],
            "query": state["current_query"],
            "result": state["retriever_result"]
        })
        
        return {
            **state,
            "suggested_tool": suggestion.get("tool"),
            "tool_input": suggestion.get("input"),
            "requires_confirmation": bool(suggestion.get("tool")),
            "confirmed": False
        }
        
    def _execute_tool_with_checkpoint(self, state: AgentState) -> AgentState:
        """Execute tool with tracking and checkpointing"""
        try:
            # Load checkpoint if exists
            checkpointer = state.get('config', {}).get('checkpointer')
            if checkpointer:
                checkpoint = checkpointer.get_checkpoint(f"execute_{state['state_id']}")
                if checkpoint:
                    return checkpoint

            # Track tool execution
            execution_record = {
                'tool': state.get('suggested_tool'),
                'input': state.get('tool_input'),
                'timestamp': time.time(),
                'state_id': state['state_id']
            }

            # Execute tool
            result = self._execute_tool(state)

            # Update execution record
            execution_record.update({
                'status': 'success' if 'error' not in result else 'error',
                'result': result.get('tool_result'),
                'error': result.get('error'),
                'completion_time': time.time()
            })

            # Update execution history
            result['execution_history'] = result.get('execution_history', []) + [execution_record]

            # Save checkpoint
            if checkpointer:
                checkpointer.save_checkpoint(f"execute_{state['state_id']}", result)
                result['last_checkpoint'] = time.time()

            return result
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return {**state, 'error': str(e)}

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
                "final_response": f"Sorry, the tool {tool_name} is not available."
            }
            
        try:
            # Execute tool with current state context
            result = tool.run(
                query=state["current_query"],
                chat_history=state.get("messages", []),
                retriever_result=state.get("retriever_result")
            )
            
            return {
                **state,
                "tool_result": result,
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
        
        return {
            **state,
            "tool_result": result,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Executed {tool} with result: {result}"}
            ]
        }
        
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response combining retriever and tool results"""
        if state.get("error"):
            return {
                **state,
                "final_response": f"I encountered an error: {state['error']}"
            }
            
        if state.get("requires_confirmation") and not state.get("confirmed"):
            tool_desc = state.get("tool_description", "this action")
            return {
                **state,
                "final_response": f"Would you like me to {tool_desc}? Please confirm."
            }
            
        # Create response prompt based on state
        prompt_parts = [
            "Based on the following information, provide a helpful response:",
            f"Query: {state['current_query']}",
            f"Retrieved Context: {state.get('retriever_result', 'No context available')}"
        ]
        
        if state.get("tool_result"):
            prompt_parts.append(f"Tool Result: {state['tool_result']}")
            
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
        
        return {
            **state,
            "final_response": response,
            "messages": state["messages"] + [
                {"role": "assistant", "content": response}
            ]
        }
            
    def _analyze_for_tool_suggestion(self, query: str, retriever_response: Optional[str]) -> Optional[Dict]:
        """Analyze if query suggests using a specific tool"""
        tool_patterns = {
            'scene_analysis': {
                'patterns': ['analyze scene', 'what is happening', 'describe scene', 'what do you see'],
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

    def _route_after_suggestion(self, state: AgentState) -> str:
        """Route to next node after tool suggestion"""
        if state.get("error"):
            return "respond"
        if state["requires_confirmation"]:
            return "execute" if state["confirmed"] else "respond"
        return "respond"
        
    def _route_after_execution(self, state: AgentState) -> str:
        """Route to next node after tool execution"""
        # Check if we need to suggest another tool
        if state.get("tool_result", {}).get("suggest_next_tool"):
            return "suggest"
        return "respond"
        
    def process_query(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """Process user query through the workflow"""
        # Initialize state
        state = AgentState(
            messages=chat_history or [],
            current_query=query,
            retriever_result=None,
            suggested_tool=None,
            tool_input=None,
            requires_confirmation=False,
            confirmed=False,
            final_response=None
        )
        
        # Run workflow
        result = self.workflow.invoke(state)
        
        return {
            "response": result["final_response"],
            "suggested_tool": result.get("suggested_tool"),
            "requires_confirmation": result.get("requires_confirmation", False),
            "chat_history": result["messages"]
        }
            
