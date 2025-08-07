
## ✅ Final Developer Prompt: Modular Streamlit App for Cortex Analyst & SQL Operator

### 🎯 Objective:

Build a clean, modular Streamlit app (that runs inside or outside Snowflake) with:

* Multi-agent chat interface
* Sidebar for **query selection**
* Modular functions
* Tabbed **data visualization**
* SQL Operator integration

---

### 📦 App Features & Structure

---

#### 1. **Sidebar: Semantic Model & Query Selector**

* Select semantic model to pass to **Cortex Analyst Agent**
* Show only **Cortex Analyst SQL queries** in **dropdown (select only one)**
* ✳️ Remove all clutter: *Connection status*, *Agent activity*, *Recent activity*, etc.

---

#### 2. **Main Chat Area (Bottom-fixed Text Input)**

* Chat interface for interacting with **Cortex Analyst** and other agents
* Text area fixed at **bottom of screen**
* Render messages in order by agent
* Maintain multi-agent context

---

#### 3. **SQL Operator Execution Panel**

* On query selection from sidebar, execute via **SQL Operator Agent**
* Display result in dedicated panel below the query selector

---

#### 4. **Data Visualization Panel (Tab-based)**

* Use **tabs** to show:

  * **Line Chart**
  * **Bar Chart**
* Use `st.line_chart`, `st.bar_chart`, or `altair` depending on data

---

### 🧩 Modular Function Structure

Structure your app as:

```python
# streamlit_app.py
def sidebar_module():
    # Handles semantic model and SQL dropdown
    pass

def chat_module():
    # Multi-agent chat interface, textarea at bottom
    pass

def cortex_analyst_module(selected_model):
    # Cortex Analyst logic to return SQL + responses
    pass

def sql_operator_module(selected_query):
    # Executes selected query and returns result
    pass

def visualization_module(sql_result_df):
    # Tabbed visualization using Streamlit tabs
    pass

def main():
    # App layout and coordination
    sidebar_module()
    chat_module()
    if selected_query:
        sql_result = sql_operator_module(selected_query)
        visualization_module(sql_result)

if __name__ == "__main__":
    main()
```

---

### 🧼 UX/UI Requirements

* ✅ Text area always pinned to bottom of chat
* ✅ Sidebar: clean — only show:

  * Semantic model selector
  * SQL queries from Cortex Analyst (dropdown, single-select)
* ✅ Use `st.chat_message` for chat clarity
* ✅ Tabs: Bar & Line chart (auto-detect column)
* ✅ Avoid exposing any backend or system messages
