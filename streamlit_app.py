import streamlit as st
import pandas as pd
import altair as alt
from typing import List, Dict, Any, Optional
import re
import json
import os
from dotenv import load_dotenv

# Import our custom modules
from multi_agent_workflow import run_workflow
from chatsnowflakecortex_wrapper import ChatSnowflakeCortex

load_dotenv()

# App Configuration
APP_CONFIG = {
    "page_title": "Cortex Analyst & SQL Operator",
    "page_icon": "🤖",
    "layout": "wide",
    "sidebar_state": "expanded"
}

SEMANTIC_MODELS = {
    "Sales Intelligence": {
        "path": "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml",
        "description": "Sales performance, revenue, and customer metrics",
        "tables": ["sales_metrics"]
    },
}

DEFAULT_SEMANTIC_MODEL = os.getenv(
    "SEMANTIC_MODELS", 
    "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml"
)

DATABASE_CONFIG = {
    "database": os.getenv("SNOWFLAKE_DATABASE", "SALES_INTELLIGENCE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "DATA"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
}

# Agent Configuration
AGENT_CONFIG = {
    "default_model": os.getenv("DEFAULT_MODEL", "llama3.1-70b"),
    "agent_model": os.getenv("AGENT_MODEL", "claude-3-5-sonnet"),
    "temperature": 0.1,
    "max_tokens": 1000,
    "timeout": int(os.getenv("API_TIMEOUT", "50000"))
}

# UI Configuration
UI_CONFIG = {
    "chat_max_height": "80vh",
    "sidebar_width": 350,
    "chart_width": 600,
    "chart_height": 400,
    "max_query_display_length": 50,
    "max_chart_rows": 20
}

def get_semantic_models() -> Dict[str, Dict]:
    """Get available semantic models"""
    return SEMANTIC_MODELS

def get_semantic_model_paths() -> List[str]:
    return [model["path"] for model in SEMANTIC_MODELS.values()]

def get_database_config() -> Dict[str, str]:
    """Get database connection configuration"""
    return DATABASE_CONFIG

def get_agent_config() -> Dict:
    """Get agent configuration"""
    return AGENT_CONFIG

def get_ui_config() -> Dict:
    """Get UI configuration"""
    return UI_CONFIG

# Configure Streamlit page with responsive settings
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["sidebar_state"]
)

def extract_sql_queries(text: str) -> List[str]:
    """
    Extract SQL queries from analyst agent response.
    Simply extracts all SQL code blocks and SQL-like statements.
    """
    sql_queries = []
    code_block_pattern = r'```(?:sql)?\s*(.*?)```'
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for block in code_blocks:
        block = block.strip()
        if block:
            sql_queries.append(block)
            
    
    return sql_queries

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_semantic_model" not in st.session_state:
        # Use first semantic model as default
        default_models = get_semantic_model_paths()
        st.session_state.selected_semantic_model = default_models[0] if default_models else ""
    
    if "available_sql_queries" not in st.session_state:
        st.session_state.available_sql_queries = []
    
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = ""
    
    if "sql_result_df" not in st.session_state:
        st.session_state.sql_result_df = None
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_session"
    
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

def semantic_model_header():
    """Semantic model selector at the top of the page"""
    st.header("DM Assistant - Data Insights & SQL Operations")
    
    semantic_models = get_semantic_models()
    model_names = list(semantic_models.keys())
    model_paths = [semantic_models[name]["path"] for name in model_names]
    
    # Model selector with descriptions
    selected_model_index = st.selectbox(
        "Choose Semantic Model:",
        options=range(len(model_names)),
        format_func=lambda i: f"{model_names[i]} - {semantic_models[model_names[i]]['description'][:50]}...",
        index=0,
        key="semantic_model_selector"
    )
    
    selected_model_path = model_paths[selected_model_index]
    st.session_state.selected_semantic_model = selected_model_path
    
    st.markdown("---")
    return selected_model_path

def sidebar_module():
    st.sidebar.title("Configuration")
    
    return st.session_state.selected_query

def chat_module():
    # Create a responsive scrollable container for messages
    # Use a reasonable pixel height that works well across devices
    chat_container = st.container(height=500)
    
    with chat_container:
        if st.session_state.messages:
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                
                display_role = role
                if role == "analyst":
                    display_role = "🔍 Cortex Analyst"
                elif role == "sqloperator":
                    display_role = "⚙️ SQL Operator"
                elif role == "general":
                    display_role = "💬 General Assistant"
                elif role == "router":
                    continue  # Skip router messages from display
                
                with st.chat_message(role):
                    if role == "user":
                        st.markdown(f"**You:** {content}")
                    else:
                        st.markdown(f"**{display_role}:**")
                        st.markdown(content)
            
            st.empty()
        else:
            st.info("Welcome! Ask me anything about your data or request help to get started.")
    
    user_input = st.chat_input("Ask me anything about your data or request help...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.awaiting_response = True
        
        st.rerun()
    
    if st.session_state.awaiting_response:
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("🤖 Processing your request..."):
                    last_user_message = None
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "user":
                            last_user_message = msg["content"]
                            break
                    
                    if last_user_message:
                        agent_config = get_agent_config()
                        result = run_workflow(last_user_message, st.session_state.thread_id)
                        
                        if result:
                            agent_role = result.get("agent_type", "general")
                            response_content = result.get("final_response", "No response available")
                            print(f"Agent Role: {agent_role}, Response: {response_content}")
                            
                            # Extract SQL queries from response
                            sql_queries = extract_sql_queries(response_content)
                            st.session_state.available_sql_queries.extend(sql_queries)
                            
                            # Append the agent response to the chat
                            st.session_state.messages.append({
                                "role": agent_role, 
                                "content": response_content
                            })
                            
                            # if result.get("sql_queries"):
                            #     st.session_state.available_sql_queries = result["sql_queries"]
                                
        
        # Clear the awaiting response flag
        st.session_state.awaiting_response = False
        
        # Rerun to show the agent response
        st.rerun()

def cortex_analyst_module(selected_model: str) -> Dict[str, Any]:
    
    return {
        "text": "",
        "sql": "",
        "citations": [],
        "formatted_response": ""
    }

def sql_operator_module(selected_query: str) -> Optional[pd.DataFrame]:
    """Executes selected query and returns result"""
    if not selected_query or not selected_query.strip():
        return None
    
    try:
        agent_config = get_agent_config()
        llm = ChatSnowflakeCortex(
            model=agent_config["default_model"],
            temperature=agent_config["temperature"],
            max_tokens=agent_config["max_tokens"],
        )
        
        if llm.session:
            cursor = llm.session.cursor()
            cursor.execute(selected_query)
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            df = pd.DataFrame(results, columns=columns)
            cursor.close()
            
            return df
        else:
            st.error("Snowflake connection not available")
            return None
            
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

def visualization_module(sql_result_df: pd.DataFrame):
    if sql_result_df is None or sql_result_df.empty:
        st.info("No data to visualize")
        return
    
    ui_config = get_ui_config()
    st.subheader("Report Visualization")
    
    data_tab, line_tab, bar_tab = st.tabs(["📋 Data", "📈 Line Chart", "📊 Bar Chart"])
    
    with data_tab:
        
        st.subheader("Report Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(sql_result_df))
        with col2:
            st.metric("Total Columns", len(sql_result_df.columns))
        with col3:
            numeric_cols = sql_result_df.select_dtypes(include=['number']).columns
            st.metric("Numeric Columns", len(numeric_cols))

        st.dataframe(sql_result_df, use_container_width=True, hide_index=True)
    
    with line_tab:
        if len(sql_result_df.columns) >= 2:
            # Auto-detect columns for line chart
            numeric_cols = sql_result_df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 1:
                if len(sql_result_df.columns) > len(numeric_cols):
                    non_numeric_cols = sql_result_df.select_dtypes(exclude=['number']).columns.tolist()
                    x_col = non_numeric_cols[0] if non_numeric_cols else sql_result_df.columns[0]
                else:
                    x_col = sql_result_df.columns[0]
                
                # Create line chart
                try:
                    if x_col in numeric_cols:
                        chart_data = sql_result_df[[x_col] + [col for col in numeric_cols if col != x_col][:3]]  # Limit to 3 y-axes
                        st.line_chart(chart_data.set_index(x_col))
                    else:
                        chart = alt.Chart(sql_result_df.head(ui_config["max_chart_rows"])).mark_line(point=True).encode(
                            x=alt.X(x_col, title=x_col),
                            y=alt.Y(numeric_cols[0], title=numeric_cols[0]),
                            tooltip=list(sql_result_df.columns)
                        ).properties(
                            width=ui_config["chart_width"],
                            height=ui_config["chart_height"],
                            title="Line Chart Visualization"
                        )
                        st.altair_chart(chart, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Could not create line chart: {str(e)}")
                    st.info("Data preview:")
                    st.dataframe(sql_result_df.head())
            else:
                st.info("No numeric columns available for line chart")
        else:
            st.info("Need at least 2 columns for line chart")
    
    with bar_tab:
        if len(sql_result_df.columns) >= 2:
            numeric_cols = sql_result_df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 1:
                if len(sql_result_df.columns) > len(numeric_cols):
                    non_numeric_cols = sql_result_df.select_dtypes(exclude=['number']).columns.tolist()
                    x_col = non_numeric_cols[0] if non_numeric_cols else sql_result_df.columns[0]
                else:
                    x_col = sql_result_df.columns[0]
                
                try:
                    if x_col in numeric_cols:
                        chart_data = sql_result_df[[x_col]].head(ui_config["max_chart_rows"])
                        st.bar_chart(chart_data)
                    else:
                        if len(numeric_cols) >= 1:
                            y_col = numeric_cols[0]
                            agg_data = sql_result_df.groupby(x_col)[y_col].sum().reset_index()
                            
                            chart = alt.Chart(agg_data.head(ui_config["max_chart_rows"])).mark_bar().encode(
                                x=alt.X(x_col, title=x_col, sort='-y'),
                                y=alt.Y(y_col, title=y_col),
                                tooltip=[x_col, y_col]
                            ).properties(
                                width=ui_config["chart_width"],
                                height=ui_config["chart_height"],
                                title="Bar Chart Visualization"
                            )
                            st.altair_chart(chart, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Could not create bar chart: {str(e)}")
                    st.info("Data preview:")
                    st.dataframe(sql_result_df.head())
            else:
                st.info("No numeric columns available for bar chart")
        else:
            st.info("Need at least 2 columns for bar chart")

def sql_queries_panel():
    st.subheader("Reports / SQL Queries")
    
    if st.session_state.available_sql_queries:
        ui_config = get_ui_config()
        max_length = ui_config["max_query_display_length"]
        
        query_options = ["Select a query..."] + [
            f"{i+1}: {query[:max_length]}..." if len(query) > max_length else f"{i+1}: {query}"
            for i, query in enumerate(st.session_state.available_sql_queries)
        ]
        
        selected_index = st.selectbox(
            "Available SQL queries from Cortex Analyst:",
            options=range(len(query_options)),
            format_func=lambda i: query_options[i],
            index=0, 
            key="sql_query_selector"
        )
        
        if selected_index > 0: 
            st.session_state.selected_query = st.session_state.available_sql_queries[selected_index - 1]
        else:
            st.session_state.selected_query = ""
    else:
        st.info("Ask Cortex Analyst a data question to see SQL queries here!")

def sql_execution_panel():
    if st.session_state.selected_query:        
        st.code(st.session_state.selected_query, language="sql")
        
        if st.button("Execute Query", type="primary"):
            with st.spinner("Executing SQL query..."):
                result_df = sql_operator_module(st.session_state.selected_query)
                st.session_state.sql_result_df = result_df
                
                if result_df is not None:
                    st.success(f"Query executed successfully! Retrieved {len(result_df)} rows.")
                else:
                    st.error("Query execution failed")

def main():
    """Main app layout and coordination"""
    initialize_session_state()
    
    # Semantic model selector at the top
    selected_model = semantic_model_header()
    
    # Sidebar module for additional configuration
    selected_query = sidebar_module()
    
    if st.session_state.get('mobile_view', False):
        chat_module()
        st.markdown("---")
        sql_queries_panel()
        if selected_query:
            sql_execution_panel()
            if st.session_state.sql_result_df is not None:
                visualization_module(st.session_state.sql_result_df)
    else:
        col1, col2 = st.columns([3, 2])  
        
        with col1:
            chat_module()
        
        with col2:
            sql_queries_panel()
            
            if selected_query:
                sql_execution_panel()
                
                if st.session_state.sql_result_df is not None:
                    visualization_module(st.session_state.sql_result_df)

if __name__ == "__main__":
    main()
