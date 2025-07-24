

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState
from chatsnowflakecortex_wrapper import ChatSnowflakeCortex
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Agent names
ROUTER_AGENT = "router"
ANALYST_AGENT = "analyst"
SQL_GENERATOR_AGENT = "sql_generator"
GENERAL_AGENT = "general"

class WorkflowState(MessagesState):
    """State schema for the multi-agent workflow"""
    user_input: str
    general_response: str
    final_response: str
    agent_type: str

# Initialize the Snowflake Cortex chat model
llm = ChatSnowflakeCortex(
    model="llama3.1-70b",
    temperature=0.1,
    max_tokens=1000,
)

def router_agent(state: WorkflowState):
    """
    Router agent that classifies user requests into specific categories
    Following the exact prompt from PROMPT.MD
    """
    system_prompt = """You are a routing agent that classifies user requests into specific categories. Your job is to analyze the user's input and determine which specialized agent should handle the request.

Classification Rules:
- If the user is asking for data analysis, insights, trends, patterns, querying data, finding information from database, or interpretation of data: respond with exactly "ANALYSIS"
- If the user is asking to CREATE tables, INSERT data, or generate database schema creation statements: respond with exactly "SQLGENERATOR"
- For any other general questions, conversations, or requests: handle them normally as a general assistant

IMPORTANT: 
- Report GENERATION and RUNNING = ANALYSIS
- Report SAVING and SCHEDULING = SQLGENERATOR

IMPORTANT: When routing to specialized agents, respond with ONLY the single word (ANALYSIS or SQLGENERATOR) - no explanations, no additional text, no formatting.

Examples:
- "Can you analyze the sales trends for last quarter?" → ANALYSIS
- "Show me the pattern in customer behavior" → ANALYSIS
- "Find all customers who purchased in the last 30 days" → ANALYSIS
- "Create a query to get top performing products" → ANALYSIS
- "Generate a query to find duplicate records" → ANALYSIS
- "Create a table for storing user information" → SQLGENERATOR
- "Insert sample data into the products table" → SQLGENERATOR
- "Generate CREATE statement for orders table" → SQLGENERATOR
- "What's the weather like today?" → [Handle as general assistant]"""

    # Get the latest user message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        user_message = state.get("user_input", "")

    # Create messages for the LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    response_content = response.content.strip().upper()
    
    # Store the response for routing logic
    state["general_response"] = response_content
    state["user_input"] = user_message
    
    # Add router response to messages
    router_message = AIMessage(content=response_content, name="Router")
    state["messages"].append(router_message)
    
    return state

def analyst_agent(state: WorkflowState):
    """
    Specialized data analyst agent using Snowflake Cortex Agent endpoint
    Following the exact prompt from PROMPT.MD
    """
    # Get the original user input
    user_input = state.get("user_input", "")
    
    try:
        # Use the Snowflake Cortex Agent endpoint for analysis
        response_content = llm.snowflake_agent_call(user_input)
        
        # Create an AI message with the formatted response
        response = AIMessage(content=response_content, name="Analyst")
        
        # Store final response and add to messages
        state["final_response"] = response_content
        state["agent_type"] = ANALYST_AGENT
        state["messages"].append(response)
        
    except Exception as e:
        # Fallback to regular LLM if Cortex Agent fails
        system_prompt = """You are a specialized data analyst agent. Your role is to provide comprehensive data analysis, insights, interpretations, and generate SELECT queries to retrieve data.

Your capabilities include:
- Analyzing data patterns and trends
- Providing statistical insights
- Interpreting data visualizations
- Identifying correlations and anomalies
- Generating analytical reports
- Writing SELECT queries to extract data for analysis
- Creating complex JOINs and subqueries for data retrieval
- Implementing aggregate functions and window functions
- Recommending data-driven decisions

When responding:
1. Be thorough and analytical in your approach
2. Use clear, professional language
3. Provide actionable insights
4. Include relevant context and explanations
5. Generate SELECT queries when data retrieval is needed
6. Suggest next steps or recommendations when appropriate

Focus on delivering high-quality analytical insights and data retrieval queries that help users understand their data better."""

        # Create messages for the analyst
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        
        response = llm.invoke(messages)
        response.name = "Analyst"
        
        # Store final response and add to messages
        state["final_response"] = f"**Fallback Response (Cortex Agent unavailable):**\n{response.content}"
        state["agent_type"] = ANALYST_AGENT
        state["messages"].append(response)
        
        if llm.debug:
            print(f"Analyst agent fell back to regular LLM due to error: {e}")
    
    return state

def sql_generator_agent(state: WorkflowState):
    """
    Specialized SQL generator agent focused on database creation and data insertion
    Following the exact prompt from PROMPT.MD
    """
    system_prompt = """You are a specialized SQL generator agent focused ONLY on database creation and data insertion operations. Your role is to create database schemas and insert data safely.

Your ALLOWED capabilities include:
- Writing CREATE TABLE statements
- Creating database schemas and table structures
- Writing INSERT statements to add data
- Creating indexes (CREATE INDEX)
- Creating views (CREATE VIEW)
- Creating stored procedures and functions
- Designing database relationships and constraints

REPORT MANAGEMENT FEATURES:
1. **Save Report**: Insert report metadata into the reporting table:
   Table: SNOWFLAKE_LEARNING_DB.TAUSEEFUSMAN_LOAD_SAMPLE_DATA_FROM_S3.REPORTING_TABLE
   Columns: USERNAME, REPORT_NAME, REPORT_QUERY, CREATED_AT, UPDATED_AT

2. **Schedule Report**: Create complete scheduling solution including:
   - New table for report output (if needed)
   - Stored procedure to execute the report query
   - Scheduled task to run the procedure automatically
   - Task management commands (RESUME/SUSPEND)

SNOWFLAKE SCHEDULING TEMPLATE:

-- 1. Create output table (if needed)
CREATE OR REPLACE TABLE schema.report_output_table AS
SELECT * FROM (your_report_query) LIMIT 0;

-- 2. Create stored procedure
CREATE OR REPLACE PROCEDURE schema.report_procedure()
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    -- Clear previous data
    DELETE FROM schema.report_output_table;
    -- Insert new data
    INSERT INTO schema.report_output_table
    SELECT * FROM (your_report_query);
    RETURN 'Report created successfully '
END;
$$;

-- 3. Create scheduled task
CREATE OR REPLACE TASK schema.report_task
  WAREHOUSE = your_warehouse
  SCHEDULE = 'USING CRON 0 0 * * * UTC' -- Daily at midnight
AS
  CALL schema.report_procedure();

-- 4. Start the task
ALTER TASK schema.report_task RESUME;

STRICTLY FORBIDDEN operations:
- DELETE statements
- DROP statements (tables, databases, indexes, etc.)
- TRUNCATE statements  
- UPDATE statements
- ALTER statements that remove data or structures
- Any destructive database operations

When generating SQL:
1. Always focus on CREATE and INSERT operations only
2. Write clean, well-formatted SQL code
3. Include proper constraints (PRIMARY KEY, FOREIGN KEY, NOT NULL, etc.)
4. Consider data types and field sizes carefully
5. Provide sample INSERT statements with realistic data
6. Include comments for complex table structures
7. If asked for destructive operations, politely decline and explain your limitations


SECURITY NOTICE: You are designed to only perform safe, non-destructive database operations to protect data integrity."""

    # Get the original user input
    user_input = state.get("user_input", "")
    
    # Create messages for the SQL generator
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    response = llm.invoke(messages)
    response.name = "SQL_Generator"
    
    # Store final response and add to messages
    state["final_response"] = response.content
    state["agent_type"] = SQL_GENERATOR_AGENT
    state["messages"].append(response)
    
    return state

def general_agent(state: WorkflowState):
    """
    General assistant for other types of requests
    """
    system_prompt = """You are a helpful general assistant. Answer the user's question in a friendly and informative manner."""

    # Get the original user input
    user_input = state.get("user_input", "")
    
    # Create messages for the general agent
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    response = llm.invoke(messages)
    response.name = "General_Assistant"
    
    # Store final response and add to messages
    state["final_response"] = response.content
    state["agent_type"] = GENERAL_AGENT
    state["messages"].append(response)
    
    return state

def route_request(state: WorkflowState) -> Literal["analyst", "sql_generator", "general"]:
    """
    Route requests based on general agent response
    Following the routing logic from PROMPT.MD
    """
    response = state.get("general_response", "").strip().upper()
    
    print(f"DEBUG: Routing based on response: '{response}'")
    
    if response == "ANALYSIS":
        print("Routing to analyst agent")
        return "analyst"
    elif response == "SQLGENERATOR":
        print("Routing to sql_generator agent")
        return "sql_generator"
    else:
        print("Routing to general agent")
        return "general"

# Build the workflow graph
def create_workflow():
    """Create and compile the multi-agent workflow graph"""
    
    # Initialize the graph builder
    builder = StateGraph(WorkflowState)
    
    # Add all agent nodes
    builder.add_node("router", router_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("sql_generator", sql_generator_agent)
    builder.add_node("general", general_agent)
    
    # Set entry point
    builder.add_edge(START, "router")
    
    # Add conditional edges from router to specialized agents
    builder.add_conditional_edges(
        "router",
        route_request,
        {
            "analyst": "analyst",
            "sql_generator": "sql_generator", 
            "general": "general"
        }
    )
    
    # All specialized agents end the conversation
    builder.add_edge("analyst", END)
    builder.add_edge("sql_generator", END)
    builder.add_edge("general", END)
    
    # Add memory for conversation state
    memory = MemorySaver()
    
    # Compile the graph
    graph = builder.compile(checkpointer=memory)
    
    return graph

# Create the workflow graph
workflow_graph = create_workflow()

def run_workflow(user_input: str, thread_id: str = "default"):
    """
    Run the multi-agent workflow with a user input
    
    Args:
        user_input: The user's question or request
        thread_id: Unique identifier for the conversation thread
        
    Returns:
        The final response from the appropriate agent
    """
    
    # Initialize state
    initial_state = WorkflowState(
        messages=[HumanMessage(content=user_input, name="User")],
        user_input=user_input,
        general_response="",
        final_response="",
        agent_type=""
    )
    
    # Configuration for the thread
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Run the workflow
        result = workflow_graph.invoke(initial_state, config)
        
        print(f"\n=== Multi-Agent Workflow Result ===")
        print(f"User Input: {user_input}")
        print(f"Routed to: {result['agent_type']}")
        print(f"Final Response: {result['final_response']}")
        print("=" * 40)
        
        return result
        
    except Exception as e:
        print(f"Error running workflow: {e}")
        return None

def interactive_demo():
    """Interactive demo to test the workflow"""
    print("🤖 Multi-Agent Workflow Demo")
    print("=" * 50)
    print("This workflow includes:")
    print("- Router Agent: Classifies your request")
    print("- Analyst Agent: Handles data analysis requests")
    print("- SQL Generator Agent: Creates tables and inserts data")
    print("- General Agent: Handles other requests")
    print("=" * 50)
    print("Type 'quit' to exit")
    print()
    
    thread_id = "demo_session"
    
    while True:
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
            
        if not user_input:
            print("Please enter a message.")
            continue
        
        # Run the workflow
        result = run_workflow(user_input, thread_id)
        
        if result:
            print(f"🤖 {result['agent_type'].title()}: {result['final_response']}")
        
        print()

if __name__ == "__main__":
    print("Multi-Agent Workflow compiled successfully!")
    
    
    # Start interactive demo
    print("\n" + "=" * 50)
    interactive_demo()

