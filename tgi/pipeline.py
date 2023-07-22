import contextlib
import gc
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from tgi.fast_init import fast_init
from tgi.metrics import Metrics
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    GPTBigCodeConfig,
)
from text_generation_server.cache import Cache
from text_generation_server.pb import generate_pb2_grpc, generate_pb2

from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
)


logger = logging.getLogger(__name__)

def parse_revision(pretrained_model: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    revision = None
    if pretrained_model is not None:
        pretrained_split = pretrained_model.split(":", 1)
        if len(pretrained_split) == 2:
            pretrained_model, revision = pretrained_split
    return pretrained_model, revision

class Pipeline:
    def __init__(
        self,
        *,
        model_type: Optional[str] = None,
        pretrained_config: Optional[str] = None,
        pretrained_model: Optional[str] = None,
        config_args: Dict[str, Any],
        tokenizer: str,
        device: torch.device,
        dtype: torch.dtype,
        fast_init: bool = True,
        trust_remote_code: bool = False,
        is_flash:bool = True,
        quantize:Optional[str] = None,
    ):
        self.global_metrics = {}
        t0 = self._get_time()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        t1 = self._get_time()

        self.device = device
        if self.device == torch.device("cuda"):
            self.device = torch.device("cuda:0")

        self.dtype = dtype
        self.is_flash = is_flash
        self.quantize = quantize
        self.is_int8 = self.dtype == torch.int8
        self.fast_init = fast_init
        self.trust_remote_code = trust_remote_code
        if self.is_int8 and self.device != torch.device("cuda:0"):
            raise ValueError(f"Model quantization not supported on device {self.device}")

        self.config = self._get_config(model_type, pretrained_config or pretrained_model, config_args)
        t2 = self._get_time()

        logger.info(f"Model configuration: {self.config}")

        if pretrained_model is None:
            self.model = self._create_model()
            if self.is_int8:
                self._reload_model()
        else:
            self.model = self._load_pretrained(pretrained_model)

        t3 = self._get_time()
        self.global_metrics[Metrics.INIT_TOKEN] = t1 - t0
        self.global_metrics[Metrics.INIT_CONFIG] = t2 - t1
        self.global_metrics[Metrics.INIT_TOTAL] = t3 - t0 
    
    def _get_time(self, synchronize=False):
        if synchronize:
            torch.cuda.synchronize()
        return time.perf_counter()


class DS_Pipeline(Pipeline):
    def __init__(self, **kwargs):
        import deepspeed

        super().__init__(**kwargs)

        if self.device != torch.device("cuda:0"):
            raise ValueError(f"Deepspeed does not support device {self.device}")

        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Deepspeed does not support dtype {self.dtype}")

        if self.config.model_type not in ("bloom", "gpt2"):
            raise ValueError(f"Deepspeed does not support model type {self.config.model_type}")

        self.model = deepspeed.init_inference(
            self.model,
            mp_size=int(os.getenv("WORLD_SIZE", "1")),
            # base_dir="./",
            dtype=self.dtype,
            replace_with_kernel_inject=True,
        )


class TG_Pipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Ignoring dtype
        self.device = torch.device(self.device)
        self.cache = Cache()
        if self.device != torch.device("cuda:0"):
            raise ValueError(f"Textgen does not support device {self.device}")

        self.config = getattr(self.model, "config", None) or self.model.model.transformer.config

    def _get_config(
        self,
        model_type: Optional[str],
        pretrained_config: Optional[str],
        config_args: Dict[str, Any],
    ) -> Optional[PretrainedConfig]:
        return None

    def _create_model(self) -> PreTrainedModel:
        raise NotImplementedError()

    def _reload_model(self):
        raise NotImplementedError()

    def _save_pretrained(self, pretrained_model: str):
        raise NotImplementedError()

    def _load_pretrained(self, pretrained_model: str):
        from text_generation_server.models import get_model,SantaCoder,FlashSantacoder

        pretrained_model, revision = parse_revision(pretrained_model)

        with fast_init(self.device) if self.fast_init else contextlib.nullcontext():
            if self.is_flash:
                return FlashSantacoder(pretrained_model,revision,self.quantize,self.trust_remote_code)
            else:
                return SantaCoder(pretrained_model,revision,self.quantize,self.trust_remote_code)

    def _generate_hf(self, inputs: Dict, max_new_tokens: int, use_cache: bool):
        raise NotImplementedError()

    def _allocate_mock_cache(self, past_key_length: int, batch_size: int):
        raise NotImplementedError()

    def get_num_parameters(self) -> int:
        return 0 
    
    def prefill(self,batch):
        generations, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)
        return next_batch
    
    def decode(self,batch):
        batches = [] 
        batch = self.cache.pop(batch.batch_id)
        if batch is None:
            raise ValueError(f"Batch ID {batch.batch_id} not found in cache.")
        batches.append(batch)

        if len(batches) == 0:
            raise ValueError("All batches are empty")

        if len(batches) > 1:
            batch = self.model.batch_type.concatenate(batches)
        else:
            batch = batches[0]

        generations, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)
        
        return generations,next_batch
        
    def _generate_textgen(
        self,
        batch,
        max_new_tokens: int,
        use_cache: bool = True,
        do_prefill: bool = True,
        breakdown_latency: bool = False,
        key_length_step: int = 1,
        ignore_oom: bool = False,
        pad_generated_tokens: float = 0,
    ):
        t0 = self._get_time(breakdown_latency)
        assert do_prefill or use_cache
        # TODO: Implement?
        assert pad_generated_tokens == 0

        input_length = max(batch.input_lengths)
        output_length = input_length + max_new_tokens
        
        #self.model.warmup(batch)

        t1 = self._get_time(breakdown_latency)
        last_time = t1
        generate_times = {}
        with torch.inference_mode():
            for key_length in range(input_length, output_length, key_length_step):
                try:
                    generated, batch = self.model.generate_token(batch)
                    t2 = self._get_time(breakdown_latency)
                    generate_times[key_length] = t2 - last_time
                    last_time = t2
                except torch.cuda.OutOfMemoryError:
                    if ignore_oom:
                        logger.warning(f"Out of memory at key length {key_length}")
                        break
                    else:
                        raise
        output_text = ["" if g.generated_text is None else g.generated_text.text for g in generated]

        metrics = {}
        if breakdown_latency:
            metrics[Metrics.LATENCY_GENERATE_START] = t1 - t0
            metrics[Metrics.LATENCY_GENERATE_BREAKDOWN] = generate_times

        return output_text, metrics

    def __call__(
        self,
        text: List[str],
        max_new_tokens: int,
        custom_generate: bool = False,
        use_cache: bool = True,
        do_prefill: bool = True,
        breakdown_latency=False,
        key_length_step: int = 1,
        ignore_oom: bool = False,
        pad_generated_tokens: float = 0,
    ) -> Tuple[List[str], Dict[str, Any]]:
        t0 = self._get_time()

        from text_generation_server.pb import generate_pb2
        from text_generation_server.models.model import Model

        model: Model = self.model

        batch_pb = generate_pb2.Batch(
            id=0,
            requests=[
                generate_pb2.Request(
                    id=i,
                    inputs=t,
                    truncate=99999,
                    parameters=generate_pb2.NextTokenChooserParameters(
                        temperature=1.0,
                        top_p=1,
                        typical_p=1,
                        do_sample=False,
                        seed=0,
                        repetition_penalty=1.0,
                        watermark=False,
                    ),
                    stopping_parameters=generate_pb2.StoppingCriteriaParameters(
                        max_new_tokens=max_new_tokens,
                        stop_sequences=None,
                        ignore_eos_token=True,
                    ),
                )
                for i, t in enumerate(text)
            ],
            size=len(text),
            max_tokens=0,  # Ignored
        )
        batch = model.batch_type.from_pb(batch_pb, self.tokenizer,self.dtype, self.device)

        batch_size = len(batch)

        # TODO: Implement
        input_length = max(batch.input_lengths)
        output_length = input_length + max_new_tokens
        
        output_text, generate_metrics = self._generate_textgen(
            batch,
            max_new_tokens,
            use_cache,
            do_prefill,
            breakdown_latency,
            key_length_step,
            ignore_oom,
            pad_generated_tokens,
        )
        t1 = self._get_time(True)

        metrics = {
            **generate_metrics,
            Metrics.BATCH_SIZE: batch_size,
            Metrics.INPUT_LENGTH: input_length,
            Metrics.OUTPUT_LENGTH: output_length,
            Metrics.TOKENS_SAMPLE: output_length - input_length,
            Metrics.TOKENS_BATCH: batch_size * (output_length - input_length),
            Metrics.LATENCY_E2E: t1 - t0,
        }

        output_text = [i + o for i, o in zip(text, output_text)]

        return output_text, metrics