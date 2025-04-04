import copy
import os
from threading import Thread
import yaml
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Optional, Dict, Any, Union
import time
import re
import json

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import retry_on_specific_exceptions, eval_logger

import signal


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    
def save_jsonl(data: Union[List[Dict[str, Any]], Dict[str, Any]], file_path: str, append: bool = False):
    """Save a list of dictionaries to a JSONL file."""
    mode = 'a' if append else 'w'
    if isinstance(data, dict):
        data = [data]
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def force_kill_thread(force_kill_time: str):
    kill = False
    while True:
        current_time = time.strftime("%H:%M")
        if current_time == force_kill_time or kill:
            print(f"Forcing process to kill at {force_kill_time}")
            os.kill(os.getpid(), signal.SIGKILL)
            kill = True
        time.sleep(20)

def litellm_completion(model, chat: bool = False, **kwargs):
    """Query LiteLLM for completion.

    Retry with back-off until they respond
    """
    if not find_spec("litellm"):
        raise Exception(
            "attempted to use 'litellm' LM type, but package `litellm` is not installed. "
            "Please install via `pip install litellm`"
        )
    else:
        import litellm
        from litellm.exceptions import OpenAIError

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        import traceback

        traceback.print_exc()

    @retry_on_specific_exceptions(
        on_exceptions=[Exception],
        max_retries=5,  
        backoff_time = 3.0,
        backoff_multiplier = 2, 
        on_exception_callback=_exception_callback,
    )
    def completion():
        # Filter out basic parameters that will be explicitly passed
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['model', 'messages', 'prompt', 'stream']}
        if chat:
            messages = kwargs.get("messages", [])
            resp = litellm.completion(
                model=model, 
                messages=messages,
                **filtered_kwargs
            )
        else:
            prompt = kwargs.get("prompt", "")
            resp = litellm.completion(
                model=model, 
                prompt=prompt, 
                **filtered_kwargs
            )

        return resp

    resp = None
    response = None
    try:
        resp = completion()
        if hasattr(resp, 'usage'):
            eval_logger.info(f"Usage: {resp.usage}")
        response = []
        for c in resp.choices:
            content = c.message.content if hasattr(c, 'message') else c.text
            if content.startswith("assistant"):
                content = content.replace("assistant", "")
            response.append(content)
        eval_logger.info(f"Response: {response}")
        
    except Exception as e:
        eval_logger.info(f"Exception raised: {str(e)}")
        eval_logger.info(f"Object: {resp}")
        eval_logger.info("Given an empty response")
        response = [""]

    return response


def remove_surrogates(text):
    cleaned_text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    return cleaned_text.strip()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        eval_logger.error(f"Error loading YAML config: {str(e)}")
        return {}


@register_model("litellm-chat-completions")
class LiteLLMChatCompletionsLM(LM):
    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",  # Model identifier with provider prefix
        api_base: str = None,
        truncate: bool = False,
        sleep_after_request: float = None,
        config_path: str = None,
        load_log_file: str = None,
        response_jsonl: str = None,
        force_kill_time: str = None,
        kill_after_n_requests: int = None,
        max_tokens: int = None,
        **kwargs,
    ) -> None:
        """
        :param model: str
            LiteLLM model identifier with provider prefix (e.g. "openai/gpt-3.5-turbo", "anthropic/claude-3-sonnet")
        :param api_base: str
            API base URL if using a custom endpoint
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        :param config_path: str
            Path to YAML configuration file containing model parameters
        """
        super().__init__()
        try:
            import litellm
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'litellm' LM type, but package `litellm` is not installed. \
    please install via `pip install litellm`",
            )
        self.model = model
        self.api_base = api_base
        self.truncate = truncate
        self.sleep_after_request = sleep_after_request
        self.model_params = {}
        self.max_tokens = max_tokens

        # Set API base if provided
        if self.api_base:
            os.environ["LITELLM_API_BASE"] = self.api_base
            
        # Load config if provided
        if config_path:
            config = load_yaml_config(config_path)
            if config and 'model_list' in config:
                for model_config in config['model_list']:
                    if self.model.split('/')[-1] in model_config.get('model_name') or model_config.get('model_name').split('/')[-1] in self.model:
                        self.model_params = model_config.get('litellm_params', {})
                        # If model is specified in litellm_params, use that instead
                        if 'model' in self.model_params:
                            self.model = self.model_params['model']
                        eval_logger.info(f"Loaded model parameters for {self.model}: {self.model_params}")
                        break

        self.fix_text = lambda x: x.strip()
        if "gemini" in self.model:
            self.fix_text = remove_surrogates

        self.response_jsonl = response_jsonl
        self.has_existing_response_jsonl = False
        self.response_history = []
        self.response_history_ids = {}
        if response_jsonl is not None:
            if os.path.exists(response_jsonl):
                self.has_existing_response_jsonl = True
                self.response_history = load_jsonl(response_jsonl)
                self.response_history_ids = {}
                for item in self.response_history:
                    self.response_history_ids[item['id']] = item
            else:
                save_jsonl(self.response_history, response_jsonl)

        if force_kill_time is not None:
            eval_logger.info(f"Force kill time: {force_kill_time}")
            time_monitoring_thread = Thread(target=force_kill_thread, args=(force_kill_time,), daemon=True)
            time_monitoring_thread.start()

        self.kill_after_n_requests = kill_after_n_requests
        if kill_after_n_requests is not None:
            eval_logger.info(f"Killing after {kill_after_n_requests} requests")
        self.sessionrequests_count = 0

        self.load_log_file = load_log_file
        self.has_log_values = False
        if load_log_file is not None:
            eval_logger.info(f"Loading log file: {load_log_file}")
            tasks = {}
            current_task = None
            current_values = []
            f = open(load_log_file, 'r')
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "Selected Tasks: " in line:
                    #extract task names from line
                    #ex: extract bluex from 2025-04-03:02:34:30,050 INFO     [__main__.py:236] Selected Tasks: ['bluex']
                    try:
                        task_part = line.split("Selected Tasks: ")[1]
                        task_list = eval(task_part)  # Safely convert string representation of list to actual list
                        if len(task_list) == 1:
                            
                            if current_task is not None and len(current_values) > 0:
                                if current_task not in tasks or len(current_values) > len(tasks[current_task]):
                                    tasks[current_task] = current_values
                            current_task = task_list[0]
                            current_values = []
                        else:
                            raise Exception(f"Expected 1 task name, got {len(task_list)}")
                    except (IndexError, SyntaxError, ValueError) as e:
                        raise Exception(f"Failed to extract task names from line: {line}. Error: {e}")
                
                
                elif current_task is not None and "Response: " in line:
                    try:
                        values = eval(line.split("Response: ")[1])
                        if isinstance(values, list) and len(values) == 1:
                            current_values.append(values[0])
                        else:
                            raise Exception(f"Expected 1 value, got {len(values)}")
                    except (IndexError, SyntaxError, ValueError) as e:
                        raise Exception(f"Failed to extract values from line: {line}. Error: {e}")
                elif current_task is not None and "Given an empty response" in line:
                    current_values.append(None)
            
            if current_task is not None and len(current_values) > 0:
                if current_task not in tasks or len(current_values) > len(tasks[current_task]):
                    tasks[current_task] = current_values
            f.close()
            if len(tasks) > 0:
                eval_logger.info(f"Found {len(tasks)} tasks in log file: {tasks.keys()}")
                self.has_log_values = True
                self.log_tasks = tasks
                        
    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
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
        curr_task = None
        curr_index = 0
        for request in requests:
            context = request
            gen_kwargs = request.args[1]

            # Generate a unique ID for this request
            custom_id = f"{context.task_name}-{context.doc_id}"
            if hasattr(context, 'repeats') and context.repeats is not None:
                custom_id += f"-{context.repeats}"
            
            is_anthropic = "anthropic/" in self.model
            
            inps = []
            data = context.ctx_data

            inps.append({"role": "system", "content": self.fix_text(data['description'])})
            for shot, ans in data['fewshots']:
                inps.append({"role": "user", "content": self.fix_text(shot)})
                inps.append({"role": "assistant", "content": self.fix_text(ans)})
            if is_anthropic:
                last_message = inps[-1]['content']
                inps[-1]['content'] = [{
                    "type": "text",
                    "text": last_message,
                    "cache_control": {"type": "ephemeral"}
                }]
            inps.append({"role": "user", "content": self.fix_text(data['example'])})

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
                #kwargs["stop"] = until
                #if "claude" in self.model or "llama" in self.model:
                if "stop" in kwargs.keys():
                    kwargs.pop("stop")
                kwargs["temperature"] = 0
                if 'openai/o3' in self.model or 'openai/o1' in self.model:
                    kwargs.pop("temperature")
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                if 'max_gen_toks' in kwargs.keys():
                    kwargs.pop("max_gen_toks")
                #kwargs["max_tokens"] = kwargs.pop("max_gen_toks", self.max_gen_toks)
            else:
                raise ValueError(
                    f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                )

            import litellm
            
            # Merge model params from config and runtime kwargs
            # Start with basic parameters
            litellm_kwargs = {}
            
            # Add model-specific parameters from config
            if self.model_params:
                model_params_copy = self.model_params.copy()
                model_params_copy.pop('model', None)  # Remove model if present
                litellm_kwargs.update(model_params_copy)
            
            # Add runtime parameters (will override config if there are conflicts)
            litellm_kwargs.update(kwargs)

            if curr_task != context.task_name:
                curr_task = context.task_name
                curr_index = 0
            else:
                curr_index += 1
            
            use_log_condition = self.has_log_values and curr_task in self.log_tasks and curr_index < len(self.log_tasks[curr_task]) and self.log_tasks[curr_task][curr_index] is not None

            if self.has_existing_response_jsonl and custom_id in self.response_history_ids:
                response = self.response_history_ids[custom_id]['response']
                eval_logger.info(f"Using existing response for {custom_id}")
            elif use_log_condition:
                response = [self.log_tasks[curr_task][curr_index]]
                eval_logger.info(f"Using logged value at index {curr_index}")
                eval_logger.info(f"Response: {response}")

                eval_logger.info(f"Response: {response}")
            else:
                response = litellm_completion(
                    model=self.model,
                    chat=True,
                    messages=inps,
                    **litellm_kwargs,
                )
                self.sessionrequests_count += 1
                if self.kill_after_n_requests is not None and self.sessionrequests_count >= self.kill_after_n_requests:
                    eval_logger.info(f"Killing after {self.sessionrequests_count} requests")
                    os.kill(os.getpid(), signal.SIGKILL)
            
            # Extract response from litellm response format
            data = {
                'id': custom_id,
                'task': curr_task,
                'index': curr_index,
                'response': response,
            }
            if self.response_jsonl is not None and custom_id not in self.response_history_ids:
                save_jsonl(data, self.response_jsonl, append=True)
            self.response_history_ids[custom_id] = data

            s = response[0] if response else ""
            s = "" if s is None or not isinstance(s, str) else s
            

            #if until is not None:
            #    for term in until:
            #        if len(term) > 0:
            #            s = s.split(term)[0]

            res.append(s)

            self.cache_hook.add_partial(
                "generate_until", (context, {"until": until}), s
            )
            pbar.update(1)
            
            if self.sleep_after_request is not None:
                time.sleep(self.sleep_after_request)

        pbar.close()

        return res

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.") 