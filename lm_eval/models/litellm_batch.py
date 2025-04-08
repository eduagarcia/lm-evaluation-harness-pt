import copy
import os
import json
import time
import asyncio
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Optional, Dict, Any, Union, Tuple
import re

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import retry_on_specific_exceptions, eval_logger


def remove_surrogates(text):
    """Clean text by removing surrogate characters."""
    cleaned_text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    return cleaned_text.strip()


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text string."""
    try:
        import tiktoken
    except ImportError:
        eval_logger.warning("tiktoken not installed, returning estimated token count")
        return len(text.split())
    
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        eval_logger.warning(f"Model '{model}' not found in tiktoken. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))


def count_message_tokens(messages: List[Dict[str, str]], model: str) -> Tuple[int, Dict[str, int]]:
    """
    Count the number of tokens in a list of messages for chat models.
    Returns the total count and a breakdown by role.
    """
    try:
        import tiktoken
    except ImportError:
        eval_logger.warning("tiktoken not installed, returning estimated token count")
        return sum(len(m.get("content", "").split()) for m in messages), {}
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        eval_logger.warning(f"Model '{model}' not found in tiktoken. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    
    # Format msg based on OpenAI's documentation
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    tokens_per_message = 3  # every message follows <im_start>{role/name}\n{content}<im_end>\n
    tokens_per_name = 1     # if there's a name, the role is omitted
    
    # Token count by role
    token_counts = {"system": 0, "user": 0, "assistant": 0, "function": 0, "total": 0}
    
    total_tokens = 0
    for message in messages:
        num_tokens = tokens_per_message
        for key, value in message.items():
            if key == "role":
                role = value
                continue
                
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
        
        # Add tokens to role-specific count
        if role in token_counts:
            token_counts[role] += num_tokens
        
        total_tokens += num_tokens
    
    # Add 3 tokens for the assistant's reply format
    total_tokens += 3
    token_counts["total"] = total_tokens
    
    return total_tokens, token_counts


async def create_batch_file(input_file_path, custom_llm_provider):
    """Create a file for batch processing using LiteLLM."""
    try:
        import litellm
    except ImportError:
        raise Exception("Package `litellm` is not installed. Please install via `pip install litellm`")
    
    file_obj = await litellm.acreate_file(
        file=open(input_file_path, "rb"),
        purpose="batch",
        custom_llm_provider=custom_llm_provider,
    )
    eval_logger.info(f"Created batch file: {file_obj}")
    return file_obj


async def create_batch_request(input_file_id, custom_llm_provider, model, completion_window="24h"):
    """Create a batch request using LiteLLM."""
    try:
        import litellm
    except ImportError:
        raise Exception("Package `litellm` is not installed. Please install via `pip install litellm`")
    
    create_batch_response = await litellm.acreate_batch(
        completion_window=completion_window,
        endpoint="/v1/chat/completions",
        input_file_id=input_file_id,
        custom_llm_provider=custom_llm_provider,
        model=model,
        metadata={"source": "lm-evaluation-harness"},
    )
    eval_logger.info(f"Created batch request: {create_batch_response}")
    return create_batch_response


async def retrieve_batch(batch_id, custom_llm_provider):
    """Retrieve a batch using LiteLLM."""
    try:
        import litellm
    except ImportError:
        raise Exception("Package `litellm` is not installed. Please install via `pip install litellm`")
    
    retrieved_batch = await litellm.aretrieve_batch(
        batch_id=batch_id, 
        custom_llm_provider=custom_llm_provider
    )
    eval_logger.info(f"Retrieved batch: {retrieved_batch}")
    return retrieved_batch


async def get_file_content(file_id, custom_llm_provider):
    """Get file content using LiteLLM."""
    try:
        import litellm
    except ImportError:
        raise Exception("Package `litellm` is not installed. Please install via `pip install litellm`")
    
    file_content = await litellm.afile_content(
        file_id=file_id, 
        custom_llm_provider=custom_llm_provider
    )
    return file_content


def save_batch_info(batch_info, filename="litellm_batch_info.json"):
    """Save batch information to a file."""
    with open(filename, "w") as f:
        json.dump(batch_info, f, indent=2)
    eval_logger.info(f"Saved batch info to {filename}")


def load_batch_info(filename="litellm_batch_info.json"):
    """Load batch information from a file."""
    try:
        with open(filename, "r") as f:
            batch_info = json.load(f)
        return batch_info
    except FileNotFoundError:
        eval_logger.warning(f"Batch info file {filename} not found")
        return {}
    except json.JSONDecodeError:
        eval_logger.error(f"Error decoding JSON from {filename}")
        return {}


@register_model("litellm-batch-completions")
class LiteLLMBatchCompletionsLM(LM):
    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",  # Model identifier with provider prefix
        api_base: str = None,
        truncate: bool = False,
        batch_file: str = "litellm_batch_input.jsonl",
        batch_info_file: str = "litellm_batch_info.json",
        custom_llm_provider: str = "openai",
        completion_window: str = "24h",
        write_mode: bool = True,  # True to create batch, False to read results
        tokens_summary_file: str = "token_usage_summary.json",  # File to save token usage statistics
        test_mode: bool = False,
        max_tokens: int = None,
        **kwargs,
    ) -> None:
        """
        :param model: str
            LiteLLM model identifier with provider prefix
        :param api_base: str
            API base URL if using a custom endpoint
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        :param batch_file: str
            Path to the JSONL file for batch inputs
        :param batch_info_file: str
            Path to the JSON file to store batch information
        :param custom_llm_provider: str
            LiteLLM provider (e.g., "openai", "azure", "anthropic")
        :param completion_window: str
            Time window for batch completion (e.g., "24h")
        :param write_mode: bool
            If True, create batch requests; if False, read batch results
        :param tokens_summary_file: str
            Path to JSON file to store token usage statistics
        """
        super().__init__()
        try:
            import litellm
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'litellm' LM type, but package `litellm` is not installed. "
                "please install via `pip install litellm`",
            )
        
        # Try to import tiktoken for token counting
        try:
            import tiktoken
            self.has_tiktoken = True
        except ImportError:
            eval_logger.warning("tiktoken not installed. Install with 'pip install tiktoken' for accurate token counting.")
            self.has_tiktoken = False
        
        self.model = model
        # Remove provider prefix if present (for token counting)
        self.model_name = model.split('/')[-1] if '/' in model else model
        self.api_base = api_base
        self.truncate = truncate
        self.batch_file = batch_file
        self.batch_info_file = batch_info_file
        self.custom_llm_provider = custom_llm_provider
        self.completion_window = completion_window
        self.write_mode = write_mode
        self.tokens_summary_file = tokens_summary_file
        self.test_mode = test_mode
        self.max_tokens = max_tokens

        # Token usage statistics
        self.token_usage = {
            "total_prompt_tokens": 0,
            "total_messages": 0,
            "total_requests": 0,
            "by_task": {},
        }
        
        # Create clean batch input file in write mode
        if write_mode:
            open(self.batch_file, 'w').close()
            self.batch_info = {}
        else:
            # Load batch data in read mode
            self.batch_info = load_batch_info(self.batch_info_file)
            self.batch_responses = {}
            
            # Load responses if batch_info contains necessary data
            if self.batch_info and "batch_id" in self.batch_info:
                try:
                    batch_data = asyncio.run(retrieve_batch(
                        self.batch_info["batch_id"], 
                        self.custom_llm_provider
                    ))
                    
                    # Process batch data into a usable format
                    if hasattr(batch_data, "batches") and batch_data.batches:
                        for item in batch_data.batches:
                            if hasattr(item, "batch_id") and item.batch_id == self.batch_info["batch_id"]:
                                file_id = item.output_file_id
                                break
                        
                        if file_id:
                            output_content = asyncio.run(get_file_content(
                                file_id, 
                                self.custom_llm_provider
                            ))
                            
                            # Parse responses into a dictionary by custom_id
                            for line in output_content.split('\n'):
                                if line.strip():
                                    try:
                                        item = json.loads(line)
                                        if "custom_id" in item and "response" in item:
                                            self.batch_responses[item["custom_id"]] = item["response"]
                                    except json.JSONDecodeError:
                                        eval_logger.error(f"Error parsing response line: {line}")
                except Exception as e:
                    eval_logger.error(f"Error retrieving batch data: {str(e)}")

        # Set API base if provided
        if self.api_base:
            os.environ["LITELLM_API_BASE"] = self.api_base

        self.fix_text = lambda x: x.strip()
        if "gemini" in self.model:
            self.fix_text = remove_surrogates

    def _finalize_batch(self):
        """Submit the batch for processing and save batch info."""
        if not self.write_mode:
            return  # Skip finalization in read mode
        
        try:

            # Save token usage statistics to separate file
            with open(self.tokens_summary_file, 'w') as f:
                json.dump(self.token_usage, f, indent=2)

            eval_logger.info(f"Token usage summary saved to {self.tokens_summary_file}")
            eval_logger.info(f"Total prompt tokens: {self.token_usage['total_prompt_tokens']}")
            
             # Create batch request
            if self.test_mode:
                eval_logger.info("Test mode enabled. Skipping batch creation.")
                return
            
            # Create batch file
            file_obj = asyncio.run(create_batch_file(
                self.batch_file, 
                self.custom_llm_provider
            ))
            
            batch_response = asyncio.run(create_batch_request(
                file_obj.id, 
                self.custom_llm_provider, 
                self.model,
                self.completion_window
            ))
            
            # Save batch information
            self.batch_info = {
                "batch_id": batch_response.id,
                "input_file_id": file_obj.id,
                "model": self.model,
                "custom_llm_provider": self.custom_llm_provider,
                "timestamp": time.time(),
                "status": batch_response.status,
                "token_usage": self.token_usage,
            }
            save_batch_info(self.batch_info, self.batch_info_file)
            
            eval_logger.info(f"Batch info saved to {self.batch_info_file}")
            eval_logger.info(f"Batch submitted successfully. ID: {batch_response.id}")
            eval_logger.info("To retrieve results later, run the evaluation with write_mode=False")
            
        except Exception as e:
            eval_logger.error(f"Error finalizing batch: {str(e)}")
            raise

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        res = []

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        for request in requests:
            context = request
            gen_kwargs = request.args[1]

            is_anthropic = "anthropic/" in self.model
            
            data = context.ctx_data
            messages = []
            messages.append({"role": "system", "content": self.fix_text(data['description'])})
            for shot, ans in data['fewshots']:
                messages.append({"role": "user", "content": self.fix_text(shot)})
                messages.append({"role": "assistant", "content": self.fix_text(ans)})
            if is_anthropic:
                last_message = messages[-1]['content']
                messages[-1]['content'] = [{
                    "type": "text",
                    "text": last_message,
                    "cache_control": {"type": "ephemeral"}
                }]
            messages.append({"role": "user", "content": self.fix_text(data['example'])})
            
            # Count tokens in messages
            prompt_tokens, token_breakdown = count_message_tokens(messages, self.model_name)
            
            # Update token usage statistics
            self.token_usage["total_prompt_tokens"] += prompt_tokens
            self.token_usage["total_messages"] += len(messages)
            self.token_usage["total_requests"] += 1
            
            task_name = context.task_name
            if task_name not in self.token_usage["by_task"]:
                self.token_usage["by_task"][task_name] = {
                    "prompt_tokens": 0,
                    "requests": 0,
                }
            self.token_usage["by_task"][task_name]["prompt_tokens"] += prompt_tokens
            self.token_usage["by_task"][task_name]["requests"] += 1
            
            # Print token count information
            print(f"\nToken count for {task_name} example {context.doc_id}:")
            print(f"  Total tokens: {prompt_tokens}")
            print(f"  System: {token_breakdown.get('system', 0)}, User: {token_breakdown.get('user', 0)}, Assistant: {token_breakdown.get('assistant', 0)}")
            
            # Generate a unique ID for this request
            custom_id = f"{context.task_name}-{context.doc_id}"
            if hasattr(context, 'repeats') and context.repeats is not None:
                custom_id += f"-{context.repeats}"
            
            until = None
            if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                if "do_sample" in kwargs.keys():
                    kwargs.pop("do_sample")
                if "temperature" in kwargs.keys():
                    kwargs.pop("temperature")
                if "top_k" in kwargs.keys():
                    kwargs.pop("top_k")
                if "top_p" in kwargs.keys():
                    kwargs.pop("top_p")
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [kwargs]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                        )
                    kwargs["stop"] = until
                if "stop" in kwargs.keys():
                    kwargs.pop("stop")
                kwargs["temperature"] = 0
                #if 'anthropic' in self.model:
                #    kwargs["max_tokens"] = 32
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
            else:
                raise ValueError(
                    f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                )
            
            if self.write_mode:
                # Write request to batch file
                with open(self.batch_file, "a") as f:
                    if "temperature" in kwargs.keys() and ('o3' in self.model_name or 'o1' in self.model_name):
                        kwargs.pop("temperature")
                    batch_data = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model_name,
                            "messages": messages,
                            **kwargs
                        }
                    }
                    f.write(json.dumps(batch_data) + "\n")
                
                # Return empty response in write mode
                response_text = ""
            else:
                # Retrieve response from batch results in read mode
                if custom_id in self.batch_responses:
                    response_data = self.batch_responses[custom_id]
                    choices = response_data.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        response_text = message.get("content", "")
                    else:
                        response_text = ""
                else:
                    eval_logger.warning(f"No response found for ID: {custom_id}")
                    response_text = ""
            
            # Process the response
            s = response_text
            
            res.append(s)
            self.cache_hook.add_partial(
                "generate_until", (context, {"until": until}), s
            )
            pbar.update(1)
        
        pbar.close()
        
        # In write mode, finalize the batch after processing all requests
        if self.write_mode:
            self._finalize_batch()
        
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.") 