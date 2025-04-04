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
    """Count the number of tokens in a text string using Anthropic's tokenizer."""
    return len(text) // 4
    try:
        import anthropic
    except ImportError:
        eval_logger.warning("anthropic not installed, returning estimated token count")
        return len(text.split())
    
    client = anthropic.Anthropic()
    try:
        response = client.count_tokens(text)
        return response.tokens
    except Exception as e:
        eval_logger.warning(f"Error counting tokens: {str(e)}. Using fallback estimation.")
        return len(text) // 4


def count_message_tokens(messages: List[Dict[str, str]], model: str) -> Tuple[int, Dict[str, int]]:
    """
    Count the number of tokens in a list of messages for Anthropic chat models.
    Returns the total count and a breakdown by role.
    """
    try:
        import anthropic
    except ImportError:
        eval_logger.warning("anthropic not installed, returning estimated token count")
        return sum(len(m.get("content", "").split()) for m in messages), {}
    
    client = anthropic.Anthropic()
    
    # Format message content
    formatted_messages = []
    system_content = ""
    for message in messages:
        if message["role"] == "system":
            system_content = message["content"]
            continue
        elif message["role"] == "user":
            formatted_messages.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": message["content"]})
    
    # Token count by role
    token_counts = {"system": 0, "user": 0, "assistant": 0, "total": 0}
    
    # Count system tokens
    if system_content:
        try:
            system_tokens = client.count_tokens(system_content).tokens
            token_counts["system"] = system_tokens
        except Exception:
            token_counts["system"] = len(system_content) // 4
    
    # Count message tokens
    for message in formatted_messages:
        message_content = message["content"]
        if isinstance(message_content, list):
            message_content = message_content[0]
        if isinstance(message_content, dict):
            message_content = message_content.get("text", "")
        try:
            msg_tokens = client.count_tokens(message_content).tokens
            token_counts[message["role"]] += msg_tokens
        except Exception:
            token_counts[message["role"]] += len(message_content) // 4
    
    total_tokens = sum(token_counts.values())
    token_counts["total"] = total_tokens
    
    return total_tokens, token_counts


async def create_batch_request(batch_file_path, model, batch_params=None):
    """Create a batch request using Anthropic's API."""
    try:
        import anthropic
    except ImportError:
        raise Exception("Package `anthropic` is not installed. Please install via `pip install anthropic`")
    
    # Get Anthropic API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    # Set default batch parameters if not provided
    if batch_params is None:
        batch_params = {
        }
    
    # Read the JSONL input file to get requests
    with open(batch_file_path, "r") as f:
        lines = f.readlines()
    
    # Parse request objects from the JSONL file
    requests = []
    for line in lines:
        if line.strip():
            try:
                data = json.loads(line)
                requests.append(data)
            except json.JSONDecodeError as e:
                eval_logger.error(f"Error parsing request line: {line}")
                raise ValueError(f"Invalid JSON in batch file: {e}")
    
    if not requests:
        raise ValueError("No valid requests found in batch file")
    
    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    # Create batch using the beta API
    batch_response = await client.beta.messages.batches.create(
        requests=requests,
        **batch_params
    )
    
    eval_logger.info(f"Created batch request: {batch_response.id}")
    return batch_response


async def retrieve_batch(batch_id):
    """Retrieve a batch using Anthropic's API."""
    try:
        import anthropic
    except ImportError:
        raise Exception("Package `anthropic` is not installed. Please install via `pip install anthropic`")
    
    # Get Anthropic API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.AsyncAnthropic(api_key=api_key)
    retrieved_batch = await client.beta.messages.batches.retrieve(batch_id=batch_id)
    
    eval_logger.info(f"Retrieved batch: {retrieved_batch}")
    return retrieved_batch


async def get_batch_outputs(batch_id):
    """Get outputs from a completed batch using Anthropic's API."""
    try:
        import anthropic
    except ImportError:
        raise Exception("Package `anthropic` is not installed. Please install via `pip install anthropic`")
    
    # Get Anthropic API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    # Poll for batch completion
    while True:
        batch = await client.beta.messages.batches.retrieve(message_batch_id=batch_id)
        status = batch.processing_status
        
        if status == "ended":
            break
        elif status in ["failed", "canceled"]:
            raise Exception(f"Batch processing {status}")
        
        # Wait before polling again
        await asyncio.sleep(30)
    
    # Get results from the batch
    batch_outputs = {}
    results_stream = await client.beta.messages.batches.results(message_batch_id=batch_id)
    
    # Process each result
    async for entry in results_stream:
        if hasattr(entry, 'custom_id') and hasattr(entry, 'result'):
            if entry.result.type == "succeeded":
                content_blocks = entry.result.message.content
                text_content = ""
                for block in content_blocks:
                    if hasattr(block, 'text'):
                        text_content += block.text
                batch_outputs[entry.custom_id] = {"content": text_content}
    
    #print(batch_outputs)
    #time.sleep(10)
    return batch_outputs


def save_batch_info(batch_info, filename="anthropic_batch_info.json"):
    """Save batch information to a file."""
    with open(filename, "w") as f:
        json.dump(batch_info, f, indent=2)
    eval_logger.info(f"Saved batch info to {filename}")


def load_batch_info(filename="anthropic_batch_info.json"):
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


@register_model("anthropic-batch-completions")
class AnthropicBatchCompletionsLM(LM):
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: str = None,
        truncate: bool = False,
        batch_file: str = "anthropic_batch_input.jsonl",
        batch_info_file: str = "anthropic_batch_info.json",
        write_mode: bool = True,  # True to create batch, False to read results
        tokens_summary_file: str = "anthropic_token_usage_summary.json",  # File to save token usage statistics
        test_mode: bool = False,
        **kwargs,
    ) -> None:
        """
        :param model: str
            Anthropic model identifier (e.g., "claude-3-opus-20240229")
        :param api_key: str
            Anthropic API key (if not provided, will use ANTHROPIC_API_KEY env var)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        :param batch_file: str
            Path to the JSONL file for batch inputs
        :param batch_info_file: str
            Path to the JSON file to store batch information
        :param write_mode: bool
            If True, create batch requests; if False, read batch results
        :param tokens_summary_file: str
            Path to JSON file to store token usage statistics
        """
        super().__init__()
        try:
            import anthropic
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. "
                "please install via `pip install anthropic`",
            )
        
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not provided or found in environment variables")
        
        self.truncate = truncate
        self.batch_file = batch_file
        self.batch_info_file = batch_info_file
        self.write_mode = write_mode
        self.tokens_summary_file = tokens_summary_file
        self.test_mode = test_mode
        
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
                batch_outputs = asyncio.run(get_batch_outputs(self.batch_info["batch_id"]))
                self.batch_responses = batch_outputs
                eval_logger.info(f"Loaded {len(batch_outputs)} responses from batch ID: {self.batch_info['batch_id']}")
    

        # Set API key if provided
        if self.api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.api_key

        self.fix_text = lambda x: x.strip()

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
            
            # Set batch parameters
            batch_params = {
            }
            
            # Create batch request directly using the batch file path
            batch_response = asyncio.run(create_batch_request(
                self.batch_file, 
                self.model,
                batch_params
            ))
            
            # Save batch information
            self.batch_info = {
                "batch_id": batch_response.id,
                "batch_file": self.batch_file,
                "model": self.model,
                "timestamp": time.time(),
                "status": batch_response.processing_status,
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
        return 200000  # Claude 3 models have very large context windows

    @property
    def max_gen_toks(self) -> int:
        return 4096

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
            
            data = context.ctx_data
            messages = []
            
            # Format system message correctly for Anthropic
            system_message = [{
                "type": "text",
                "text": self.fix_text(data['description']),
                "cache_control": {"type": "ephemeral"}
            }]
            
            # Format few-shot examples
            for shot, ans in data['fewshots']:
                messages.append({"role": "user", "content": self.fix_text(shot)})
                messages.append({"role": "assistant", "content": self.fix_text(ans)})
            
            last_message = messages[-1]['content']
            messages[-1]['content'] = [{
                "type": "text",
                "text": last_message,
                "cache_control": {"type": "ephemeral"}
            }]
            
            # Add the actual query
            messages.append({"role": "user", "content": self.fix_text(data['example'])})
            
            # Count tokens in messages
            prompt_tokens, token_breakdown = count_message_tokens(messages, self.model)
            
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
                    #kwargs["stop_sequences"] = until
                if "stop" in kwargs.keys():
                    kwargs.pop("stop")
                if "stop_sequences" in kwargs.keys():
                    kwargs.pop("stop_sequences")
                kwargs["temperature"] = 0
                kwargs["max_tokens"] = 32
            else:
                raise ValueError(
                    f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                )
            
            if self.write_mode:
                # Write request to batch file
                with open(self.batch_file, "a") as f:
                    # Format request according to Anthropic's Message Batches API requirements
                    batch_data = {
                        "custom_id": custom_id,
                        "params": {
                            "model": self.model,
                            "max_tokens": kwargs.get("max_tokens", 32),
                            "messages": messages,
                            "system": system_message,
                            "temperature": kwargs.get("temperature", 0),
                        }
                    }
                    
                    # Add stop sequences if specified
                    if "stop_sequences" in kwargs:
                        batch_data["params"]["stop_sequences"] = kwargs["stop_sequences"]
                        
                    f.write(json.dumps(batch_data) + "\n")
                
                # Return empty response in write mode
                response_text = ""
            else:
                # Retrieve response from batch results in read mode
                if custom_id in self.batch_responses:
                    response_data = self.batch_responses[custom_id]
                    response_text = response_data.get("content", "")
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