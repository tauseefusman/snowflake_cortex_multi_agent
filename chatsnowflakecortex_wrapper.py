import json
import requests
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import Field, SecretStr, model_validator


import os
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_ROLES: List[str] = [
    "system",
    "user",
    "assistant",
]

# API Constants - loaded from environment variables
API_ENDPOINT = os.getenv("API_ENDPOINT", "/api/v2/cortex/inference:complete")
AGENT_API_ENDPOINT = os.getenv("AGENT_API_ENDPOINT", "/api/v2/cortex/agent:run")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "50000"))  # in milliseconds

# Cortex Agent Constants - loaded from environment variables
CORTEX_SEARCH_SERVICES = os.getenv("CORTEX_SEARCH_SERVICES", "sales_intelligence.data.sales_conversation_search")
SEMANTIC_MODELS = os.getenv("SEMANTIC_MODELS", "@sales_intelligence.data.models/sales_metrics_model.yaml")



class ChatSnowflakeCortexError(Exception):
    """Error with Snowpark client."""




def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {
        "content": message.content,
    }

    # Populate role and additional message data
    if isinstance(message, ChatMessage) and message.role in SUPPORTED_ROLES:
        message_dict["role"] = message.role
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _truncate_at_stop_tokens(
    text: str,
    stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text



class ChatSnowflakeCortex(BaseChatModel):
    """Snowflake Cortex based Chat model using REST API

    To use the chat model, you must have environment variables set with 
    your snowflake credentials or directly passed in as kwargs to the 
    ChatSnowflakeCortex constructor.

    This implementation uses the Snowflake Cortex REST API endpoint 
    /api/v2/cortex/inference:complete instead of SQL-based calls.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatSnowflakeCortex
            chat = ChatSnowflakeCortex()
    """

    # test_tools: Dict[str, Any] = Field(default_factory=dict)
    test_tools: Dict[str, Union[Dict[str, Any], Type, Callable, BaseTool]] = Field(
        default_factory=dict
    )

    session: Any = None
    """Snowflake connector session object used to obtain authentication token."""

    model: str = "mistral-large"
    """Snowflake cortex hosted LLM model name, defaulted to `mistral-large`.
        Refer to docs for more options. Also note, not all models support 
        agentic workflows."""

    cortex_function: str = "complete"
    """Cortex function to use, defaulted to `complete`.
        Refer to docs for more options."""

    temperature: float = 0
    """Model temperature. Value should be >= 0 and <= 1.0"""

    max_tokens: Optional[int] = None
    """The maximum number of output tokens in the response."""

    top_p: Optional[float] = 0
    """top_p adjusts the number of choices for each predicted tokens based on
        cumulative probabilities. Value should be ranging between 0.0 and 1.0. 
    """

    snowflake_username: Optional[str] = Field(default=None, alias="username")
    """Automatically inferred from env var `SNOWFLAKE_USERNAME` if not provided."""
    snowflake_password: Optional[SecretStr] = Field(default=None, alias="password")
    """Automatically inferred from env var `SNOWFLAKE_PASSWORD` if not provided."""
    snowflake_account: Optional[str] = Field(default=None, alias="account")
    """Automatically inferred from env var `SNOWFLAKE_ACCOUNT` if not provided."""
    snowflake_database: Optional[str] = Field(default=None, alias="database")
    """Automatically inferred from env var `SNOWFLAKE_DATABASE` if not provided."""
    snowflake_schema: Optional[str] = Field(default=None, alias="schema")
    """Automatically inferred from env var `SNOWFLAKE_SCHEMA` if not provided."""
    snowflake_warehouse: Optional[str] = Field(default=None, alias="warehouse")
    """Automatically inferred from env var `SNOWFLAKE_WAREHOUSE` if not provided."""
    snowflake_role: Optional[str] = Field(default=None, alias="role")
    """Automatically inferred from env var `SNOWFLAKE_ROLE` if not provided."""
    
    snowflake_token: Optional[str] = Field(default=None, alias="token")
    """Snowflake authentication token. If provided, will be used instead of username/password."""

    debug: bool = Field(default=False)
    """Enable debug logging for API requests and responses."""


    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = "auto",
        **kwargs: Any,
    ) -> "ChatSnowflakeCortex":
        """Bind tool-like objects to this chat model, ensuring they conform to
        expected formats."""

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        # self.test_tools.update(formatted_tools)
        formatted_tools_dict = {
            tool["name"]: tool for tool in formatted_tools if "name" in tool
        }
        self.test_tools.update(formatted_tools_dict)

        return self

    def snowflake_agent_call(self, query: str, limit: int = 10) -> str:
        """Call Snowflake Cortex Agent API for advanced analysis"""
        payload = {
            "model": "claude-3-5-sonnet",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ],
            "tools": [
                {
                    "tool_spec": {
                        "type": "cortex_analyst_text_to_sql",
                        "name": "analyst1"
                    }
                },
                {
                    "tool_spec": {
                        "type": "cortex_search",
                        "name": "search1"
                    }
                }
            ],
            "tool_resources": {
                "analyst1": {"semantic_model_file": SEMANTIC_MODELS},
                "search1": {
                    "name": CORTEX_SEARCH_SERVICES,
                    "max_results": limit,
                    "id_column": "conversation_id"
                }
            }
        }
        
        try:
            # Get the Agent API URL
            if not self.snowflake_account:
                raise ValueError("Snowflake account is required to construct Agent API URL")
            
            account_url = f"https://{self.snowflake_account}.snowflakecomputing.com"
            agent_api_url = f"{account_url}{AGENT_API_ENDPOINT}"
            
            if self.debug:
                print(f"DEBUG: Agent API URL: {agent_api_url}")
                print(f"DEBUG: Agent Payload: {json.dumps(payload, indent=2)}")
            
            # Make HTTP request to Snowflake Cortex Agent API
            response = requests.post(
                agent_api_url,
                headers=self._get_auth_headers(),
                json=payload,
                timeout=API_TIMEOUT / 1000,  # Convert milliseconds to seconds
                stream=True  # Enable streaming for agent responses
            )
            
            if self.debug:
                print(f"DEBUG: Agent Response Status Code: {response.status_code}")
                print(f"DEBUG: Agent Response Headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            # Process the streaming response
            response_content = self._parse_agent_streaming_response(response)
            
            # Process the response
            text, sql, citations = self._process_agent_response(response_content)
            
            # Format the response
            formatted_response = self._format_agent_response(text, sql, citations)
            return formatted_response
            
        except Exception as e:
            raise ChatSnowflakeCortexError(f"Error making Cortex Agent request: {str(e)}")
    
    def _process_agent_response(self, response) -> tuple:
        """Process response from Cortex Agent"""
        text = ""
        sql = ""
        citations = []
        
        if not response:
            return text, sql, citations
        if isinstance(response, str):
            return text, sql, citations
        
        try:
            for event in response:
                if self.debug:
                    print(f"DEBUG: Processing event: {json.dumps(event, indent=2)}")
                
                # Handle different event types - check both 'event' and 'object' fields
                event_type = event.get('event', event.get('object', ''))
                
                if event_type == "message.delta":
                    delta = event.get('delta', {})
                    
                    # Process content in delta
                    for content_item in delta.get('content', []):
                        content_type = content_item.get('type')
                        
                        if content_type == "tool_results":
                            tool_results = content_item.get('tool_results', {})
                            if 'content' in tool_results:
                                for result in tool_results['content']:
                                    if result.get('type') == 'json':
                                        json_content = result.get('json', {})
                                        new_text = json_content.get('text', '')
                                        new_sql = json_content.get('sql', '')
                                        
                                        if new_text:
                                            text += new_text
                                        if new_sql:
                                            sql = new_sql  # Replace with new SQL
                                        
                                        # Extract search results for citations
                                        search_results = json_content.get('searchResults', [])
                                        for search_result in search_results:
                                            citations.append({
                                                'source_id': search_result.get('source_id', ''), 
                                                'doc_id': search_result.get('doc_id', '')
                                            })
                        
                        elif content_type == 'text':
                            text += content_item.get('text', '')
                
                elif event_type == "message.content":
                    # Handle complete message content
                    content_items = event.get('content', [])
                    
                    for content_item in content_items:
                        if content_item.get('type') == 'text':
                            text += content_item.get('text', '')
                
                # Also check for any direct text content in the event
                if 'content' in event and isinstance(event['content'], str):
                    text += event['content']
                
                # Check for message content at top level
                if 'message' in event:
                    message = event['message']
                    if isinstance(message, dict) and 'content' in message:
                        if isinstance(message['content'], str):
                            text += message['content']
                        elif isinstance(message['content'], list):
                            for item in message['content']:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    text += item.get('text', '')
                                    
        except Exception as e:
            if self.debug:
                print(f"Error processing agent events: {str(e)}")
                import traceback
                traceback.print_exc()
            
        return text, sql, citations
    
    def _format_agent_response(self, text: str, sql: str, citations: list) -> str:
        """Format the agent response"""
        response_parts = []
        
        if text:
            # Clean up text formatting
            clean_text = text.replace("【†", "[").replace("†】", "]")
            response_parts.append(f"**Analysis Results:**\n{clean_text}")
        
        if sql:
            response_parts.append(f"\n**Generated SQL Query:**\n```sql\n{sql}\n```")
        
        if citations:
            citation_text = "\n**Citations:**"
            for i, citation in enumerate(citations, 1):
                source_id = citation.get('source_id', f'Citation {i}')
                citation_text += f"\n[{source_id}]"
            response_parts.append(citation_text)
        
        return "\n".join(response_parts) if response_parts else "No analysis results available."

    def _get_api_url(self) -> str:
        """Get the full API URL for Snowflake Cortex."""
        if not self.snowflake_account:
            raise ValueError("Snowflake account is required to construct API URL")
        
        # Construct the Snowflake URL from the account name
        account_url = f"https://{self.snowflake_account}.snowflakecomputing.com"
        return f"{account_url}{API_ENDPOINT}"

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        # Using Snowflake Token format for authentication
        
        if self.snowflake_token:
            # Use the token obtained from Snowflake session
            token = self.snowflake_token
        elif self.session:
            # Try to get token from the session if it exists
            try:
                token = self.session.rest.token
                self.snowflake_token = token  # Cache it for future use
            except Exception as e:
                raise ChatSnowflakeCortexError(f"Failed to get token from Snowflake session: {e}")
        else:
            # Try to get token from environment variable as fallback
            import os
            token = os.getenv("SNOWFLAKE_TOKEN")
            
            if not token:
                raise ValueError(
                    "No Snowflake token available. Please ensure Snowflake session is created "
                    "or set SNOWFLAKE_TOKEN environment variable."
                )
        
        return {
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
        }



    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        # Import Snowflake connector for session creation
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                """`snowflake-connector-python` package not found, please install:
                `pip install snowflake-connector-python`
                """
            )

        # Get values from input dict or environment variables
        def get_value_safe(dict_key: str, env_key: str) -> Optional[str]:
            """Safely get value from dict or environment, returning None if not found."""
            if dict_key in values and values[dict_key] is not None:
                return values[dict_key]
            import os
            env_value = os.getenv(env_key)
            return env_value

        values["username"] = get_value_safe("username", "SNOWFLAKE_USER")
        
        # Handle password separately as it needs to be a SecretStr
        password_value = get_value_safe("password", "SNOWFLAKE_PASSWORD")
        values["password"] = convert_to_secret_str(password_value) if password_value else None
        
        values["account"] = get_value_safe("account", "SNOWFLAKE_ACCOUNT") 
        values["database"] = get_value_safe("database", "SNOWFLAKE_DATABASE")
        values["schema"] = get_value_safe("schema", "SNOWFLAKE_SCHEMA")
        values["warehouse"] = get_value_safe("warehouse", "SNOWFLAKE_WAREHOUSE")
        values["role"] = get_value_safe("role", "SNOWFLAKE_ROLE")

        # Check if token is provided directly (optional)
        values["token"] = get_value_safe("token", "SNOWFLAKE_TOKEN")

        # Validate that required credentials are present (unless token is provided)
        if not values.get("token"):
            if not values["account"]:
                raise ValueError("account is required")
            if not values["username"]:
                raise ValueError("username is required")
            if not values["password"]:
                raise ValueError("password is required")

            # Create Snowflake connection to get token
            connection_params = {
                "user": values["username"],
                "password": values["password"].get_secret_value(),
                "account": values["account"],
                "role": values["role"],
                "warehouse": values["warehouse"],
                "database": values["database"],
                "schema": values["schema"],
            }

            # Remove None values from connection params
            connection_params = {k: v for k, v in connection_params.items() if v is not None}

            try:
                session = snowflake.connector.connect(**connection_params)
                values["session"] = session
                # Extract token from session for REST API authentication
                # The token is available in session.rest.token
                values["token"] = session.rest.token
            except Exception as e:
                raise ChatSnowflakeCortexError(f"Failed to create Snowflake connection: {e}")

        return values

    def __del__(self) -> None:
        """Clean up Snowflake connection when object is destroyed."""
        if getattr(self, "session", None) is not None:
            try:
                self.session.close()
            except:
                pass  # Ignore errors during cleanup

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return f"snowflake-cortex-{self.model}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Check for tool invocation in the messages and prepare for tool use
        tool_output = None
        for message in messages:
            if (
                isinstance(message.content, dict)
                and isinstance(message, SystemMessage)
                and "invoke_tool" in message.content
            ):
                tool_info = json.loads(message.content.get("invoke_tool"))
                tool_name = tool_info.get("tool_name")
                if tool_name in self.test_tools:
                    tool_args = tool_info.get("args", {})
                    tool_output = self.test_tools[tool_name](**tool_args)
                    break

        # Prepare messages for API request
        if tool_output:
            message_dicts.append(
                {"role": "system", "content": f"Tool output: {str(tool_output)}"}
            )

        # Prepare the API request payload
        payload = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
        }

        if self.debug:
            print(f"DEBUG: API URL: {self._get_api_url()}")
            print(f"DEBUG: Payload: {json.dumps(payload, indent=2)}")
            print(f"DEBUG: Headers: {self._get_auth_headers()}")

        try:
            # Make HTTP request to Snowflake Cortex API
            response = requests.post(
                self._get_api_url(),
                headers=self._get_auth_headers(),
                json=payload,
                timeout=API_TIMEOUT / 1000  # Convert milliseconds to seconds
            )
            
            if self.debug:
                print(f"DEBUG: Response Status Code: {response.status_code}")
                print(f"DEBUG: Response Headers: {dict(response.headers)}")
                print(f"DEBUG: Response Content: {response.text[:1000]}...")  # First 1000 chars
            
            response.raise_for_status()
            
            # Check if this is a streaming response (Server-Sent Events)
            if response.headers.get('content-type') == 'text/event-stream':
                # Parse the streaming response to extract the final message
                response_data = self._parse_streaming_response(response.text)
            else:
                # Regular JSON response
                response_data = response.json()
            
        except requests.exceptions.RequestException as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex API: {e}"
            )
        except json.JSONDecodeError as e:
            # Capture the actual response content for debugging
            response_content = response.text if 'response' in locals() else "No response available"
            raise ChatSnowflakeCortexError(
                f"Error parsing JSON response from Snowflake Cortex API: {e}\n"
                f"HTTP Status Code: {response.status_code if 'response' in locals() else 'Unknown'}\n"
                f"Response Content: {response_content[:500]}..."  # Truncate to first 500 chars
            )

        # Extract the AI message content from the API response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            ai_message_content = response_data["choices"][0]["message"]["content"]
        else:
            raise ChatSnowflakeCortexError("No valid response from Snowflake Cortex API")

        content = _truncate_at_stop_tokens(ai_message_content, stop)
        message = AIMessage(
            content=content,
            response_metadata=response_data.get("usage", {}),
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream_content(
        self, content: str, stop: Optional[List[str]]
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the output of the model in chunks to return ChatGenerationChunk.
        """
        chunk_size = 50  # Define a reasonable chunk size for streaming
        truncated_content = _truncate_at_stop_tokens(content, stop)

        for i in range(0, len(truncated_content), chunk_size):
            chunk_content = truncated_content[i : i + chunk_size]

            # Create and yield a ChatGenerationChunk with partial content
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_content))

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model in chunks to return ChatGenerationChunk."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Check for and potentially use a tool before streaming
        for message in messages:
            if (
                isinstance(message.content, dict)
                and isinstance(message, SystemMessage)
                and "invoke_tool" in message.content
            ):
                tool_info = json.loads(message.content.get("invoke_tool"))
                tool_name = tool_info.get("tool_name")
                if tool_name in self.test_tools:
                    tool_args = tool_info.get("args", {})
                    tool_result = self.test_tools[tool_name](**tool_args)
                    message_dicts.append(
                        {"role": "system", "content": f"Tool output: {str(tool_result)}"}
                    )

        # Prepare the API request payload for streaming
        payload = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
            "stream": True,  # Enable streaming
        }

        try:
            # Make streaming HTTP request to Snowflake Cortex API
            response = requests.post(
                self._get_api_url(),
                headers=self._get_auth_headers(),
                json=payload,
                timeout=API_TIMEOUT / 1000,  # Convert milliseconds to seconds
                stream=True
            )
            response.raise_for_status()

            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse each line as JSON (Server-Sent Events format)
                        line_data = line.decode('utf-8')
                        if line_data.startswith('data: '):
                            json_data = line_data[6:]  # Remove 'data: ' prefix
                            if json_data.strip() == '[DONE]':
                                break
                            
                            chunk_data = json.loads(json_data)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield ChatGenerationChunk(
                                        message=AIMessageChunk(content=content)
                                    )
                    except (json.JSONDecodeError, KeyError):
                        # Skip malformed chunks
                        continue

        except requests.exceptions.RequestException as e:
            # Fallback to non-streaming if streaming fails
            result = self._generate(messages, stop, run_manager, **kwargs)
            ai_message_content = result.generations[0].message.content
            
            # Stream the complete response in chunks
            for chunk in self._stream_content(ai_message_content, stop):
                yield chunk

    def _parse_streaming_response(self, response_text: str) -> dict:
        """Parse Server-Sent Events format response from Snowflake Cortex.
        
        Args:
            response_text: The raw SSE response text
            
        Returns:
            dict: Parsed response data in OpenAI-compatible format
        """
        import json
        
        # Split the response by lines and process each 'data:' line
        lines = response_text.strip().split('\n')
        
        final_content = ""
        last_response = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('data: '):
                data_json = line[6:]  # Remove 'data: ' prefix
                try:
                    chunk_data = json.loads(data_json)
                    last_response = chunk_data
                    
                    # Extract content from delta if available
                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                        choice = chunk_data['choices'][0]
                        if 'delta' in choice and 'content' in choice['delta']:
                            content = choice['delta']['content']
                            if content:
                                final_content += content
                                
                except json.JSONDecodeError:
                    # Skip invalid JSON chunks
                    continue
        
        # Build the final response in OpenAI-compatible format
        if last_response:
            return {
                "id": last_response.get("id", ""),
                "object": "chat.completion",
                "created": last_response.get("created", 0),
                "model": last_response.get("model", self.model),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": last_response.get("usage", {})
            }
        else:
            raise ChatSnowflakeCortexError("No valid data found in streaming response")
    
    def _parse_agent_streaming_response(self, response) -> list:
        """Parse streaming response from Snowflake Cortex Agent API"""
        events = []
        
        try:
            # Process streaming response line by line
            for line in response.iter_lines():
                if line:
                    line_data = line.decode('utf-8').strip()
                    
                    # Skip empty lines and comments
                    if not line_data or line_data.startswith(':'):
                        continue
                    
                    # Parse Server-Sent Events format
                    if line_data.startswith('data: '):
                        data_json = line_data[6:]  # Remove 'data: ' prefix
                        
                        # Skip [DONE] marker
                        if data_json.strip() == '[DONE]':
                            break
                        
                        try:
                            event_data = json.loads(data_json)
                            events.append(event_data)
                        except json.JSONDecodeError:
                            if self.debug:
                                print(f"DEBUG: Failed to parse event data: {data_json}")
                            continue
                            
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Error parsing streaming response: {e}")
        
        return events
