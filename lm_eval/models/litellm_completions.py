import copy
import os
import yaml
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Optional, Dict, Any
import time
import re

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import retry_on_specific_exceptions, eval_logger


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
        on_exceptions=[litellm.exceptions.OpenAIError],
        max_retries=5,  
        backoff_time = 3.0,
        backoff_multiplier = 2, 
        on_exception_callback=_exception_callback,
    )
    def completion():
        # Filter out basic parameters that will be explicitly passed
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['messages', 'prompt', 'stream', 'echo', 'max_tokens', 
                                     'temperature', 'logprobs', 'stop', 'seed']}
        
        if chat:
            messages = kwargs.get("messages", [])
            return litellm.completion(
                model=model, 
                messages=messages, 
                stream=kwargs.get("stream", False),
                max_tokens=kwargs.get("max_tokens", 0), 
                temperature=kwargs.get("temperature", 0),
                stop=kwargs.get("stop", None), 
                seed=kwargs.get("seed", None), 
                **filtered_kwargs
            )
        else:
            prompt = kwargs.get("prompt", "")
            return litellm.completion(
                model=model, 
                prompt=prompt, 
                stream=kwargs.get("stream", False),
                echo=kwargs.get("echo", False), 
                max_tokens=kwargs.get("max_tokens", 0),
                temperature=kwargs.get("temperature", 0), 
                logprobs=kwargs.get("logprobs", None),
                stop=kwargs.get("stop", None), 
                seed=kwargs.get("seed", None), 
                **filtered_kwargs
            )

    resp = None
    try:
        resp = completion()
        response = []
        for c in resp.choices:
            content = c.message.content if hasattr(c, 'message') else c.text
            if content.startswith("assistant"):
                content = content.replace("assistant", "")
            response.append(content)
        eval_logger.info(f"Response: {response}")
        
    except Exception as e:
        eval_logger.info(f"Exception raised: {str(e)}")
        eval_logger.info("Given an empty response")
        response = [""]
        # Re-raise the exception if resp is None
        if resp is None:
            raise

    return resp


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

        # Set API base if provided
        if self.api_base:
            os.environ["LITELLM_API_BASE"] = self.api_base
            
        # Load config if provided
        if config_path:
            config = load_yaml_config(config_path)
            if config and 'model_list' in config:
                for model_config in config['model_list']:
                    if model_config.get('model_name') == self.model.split('/')[-1] or model_config.get('model_name') == self.model:
                        self.model_params = model_config.get('litellm_params', {})
                        # If model is specified in litellm_params, use that instead
                        if 'model' in self.model_params:
                            self.model = self.model_params['model']
                        eval_logger.info(f"Loaded model parameters for {self.model}: {self.model_params}")
                        break

        self.fix_text = lambda x: x.strip()
        if "gemini" in self.model:
            self.fix_text = remove_surrogates

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
        for request in requests:
            context, gen_kwargs = request.args
            
            inps = []
            data = context.ctx_data
            inps.append({"role": "system", "content": self.fix_text(data['description'])})
            for shot, ans in data['fewshots']:
                inps.append({"role": "user", "content": self.fix_text(shot)})
                inps.append({"role": "assistant", "content": self.fix_text(ans)})
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
                #    if "stop" in kwargs.keys():
                kwargs.pop("stop")
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
            
            try:
                response = litellm_completion(
                    model=self.model,
                    chat=True,
                    messages=inps,
                    temperature=0,
                    **litellm_kwargs,
                )
                
                # Extract response from litellm response format
                if response.choices:
                    content = response.choices[0].message.content if hasattr(response.choices[0], 'message') else ""
                    s = content
                else:
                    s = ""
            except Exception as e:
                eval_logger.error(f"Error in litellm_completion: {str(e)}")
                s = ""  # Empty response on error

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