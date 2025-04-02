import copy
import os
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple
import time
import json

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import retry_on_specific_exceptions, eval_logger


def get_result(response, ctxlen: int) -> Tuple[float, bool]:
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = response.logprobs.token_logprobs
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(response.logprobs.token_logprobs)):
        token = response.logprobs.token_logprobs[i]
        top_tokens = response.logprobs.top_logprobs[i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


def oa_completion(client, custom_id, chat: bool = False, filename = None, write = True, data: dict = None, **kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    if not find_spec("openai") or not find_spec("tiktoken"):
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. "
            "Please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`"
        )
    else:
        import openai

    resp = None
    if write:
        if not filename:
            filename = "openai_batch_input.jsonl"
        with open(filename, "a") as f:
            data = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": kwargs,
            }
            f.write(json.dumps(data) + "\n")
    else:
        resp = data[custom_id]
            

    try:
        if resp is None:
            return [""]
        response = []
        for c in resp.choices:
            content = c.message.content
            if content.startswith("assistant"):
                content = content.replace("assistant", "")
            response.append(content)
        eval_logger.info(f"Response: {response}")
        
    except Exception as e:
        eval_logger.info(f"Exception raised: {str(e)}")
        eval_logger.info("Given an empty response")
        response = [""]

    return response

#import ftfy
import re

def remove_surrogates(text):
    cleaned_text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    return cleaned_text.strip()

@register_model("openai-chat-batch-completions", "local-chat-batch-completions")
class OpenaiChatCompletionsLM(LM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",  # GPT model or Local model using HuggingFace model paths
        base_url: str = None,
        truncate: bool = False,
        filename = None,
        write = True,
        **kwargs,
    ) -> None:
        """

        :param model: str
            Implements an OpenAI-style chat completion API for
            accessing both OpenAI OR locally-hosted models using
            HuggingFace Tokenizer
            OpenAI API model (e.g. gpt-3.5-turbo)
            using the **gen_kwargs passed on init
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        try:
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = base_url
        self.truncate = truncate
        self.write = write
        self.filename = filename
        self.data = {}
        if not write:
            with open(filename, "r") as f:
                for line in f:
                    data = json.loads(line)
                    self.data[data["custom_id"]] = data["response"]["body"]

        # Read from environment variable OPENAI_API_KEY
        # Set to EMPTY for local
        if self.base_url:
            self.client = openai.OpenAI(base_url=self.base_url, max_retries=0)
        else:
            self.client = openai.OpenAI()  # openai.AsyncOpenAI()

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
            
            id = f"{context.task_name}-{context.doc_id}"
            if context.repeats is not None:
                id += f"-{context.repeats}"
            
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
                if "claude" in self.model or "llama" in self.model:
                    if "stop" in kwargs.keys():
                        kwargs.pop("stop")
                kwargs["max_tokens"] = kwargs.pop("max_gen_toks", self.max_gen_toks)
            else:
                raise ValueError(
                    f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                )

            response = oa_completion(
                client=self.client,
                custom_id=id,
                write=self.write,
                filename=self.filename,
                chat=True,
                messages=inps,
                model=self.model,
                temperature=0,
                data=self.data,
                **kwargs,
            )

            s = response[0] if response else ""

            if until is not None:
                for term in until:
                    if len(term) > 0:
                        s = s.split(term)[0]

            res.append(s)

            self.cache_hook.add_partial(
                "generate_until", (context, {"until": until}), s
            )
            pbar.update(1)

        pbar.close()

        return res

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")