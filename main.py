import time, math
from hf_vanilla import generate as hf_generate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from hf_pipeline import init_pipeline_model, generate as hfp_generate
from tgi_pipeline_gen import generate as tgt_generate
from tgi.pipeline import TG_Pipeline
from vllm_gen import vllm_gen_generate
from vllm import LLM
import gc

# import deepspeed
# from accelerate import init_empty_weights

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_tokens", type=int, default=128, help="Number of tokens to generate"
)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--model_name", type=str, default="WizardLM/WizardCoder-15B-V1.0")
parser.add_argument(
    "--inference_engine",
    type=str,
    default=None,
    help="inference engine to use. Values can be `hf`,`tgi`,`vllm`,`hf_pipeline`",
)
parser.add_argument(
    "--input_scale_factor",
    type=int,
    default=1,
    help="Factor which to scale the input length,Used for long or short input benchmark",
)
parser.add_argument(
    "--num_cycles",
    type=int,
    default=3,
    help="Number of cycles to repeat the batch inference",
)
parser.add_argument("--no_flash", action="store_true")
parser.add_argument("--quantize", type=str, default=None)


pargs = parser.parse_args()

num_tokens = pargs.num_tokens
batch_size = pargs.batch_size
model_name = pargs.model_name
load_in_8bit = pargs.load_in_8bit
inference_engine = pargs.inference_engine
scale_input_factor = pargs.input_scale_factor
cycles = pargs.num_cycles

wb = wandb.init(project="wizcoder_benchmark", config=pargs.__dict__)

t_start = time.time()
# model_name = "bigcode/starcoder"
device = torch.device("cuda:0")
generate_kwargs = dict(
    min_new_tokens=num_tokens,
    max_new_tokens=num_tokens,
    do_sample=False,
    use_cache=True,
    temperature=0,
)

warmup_sentences = [
    "def",
] * batch_size

input_sentences = [
    "def test_",
    "def api_",
    "#DeepSpeed is a machine learning framework",
    "#He is working on",
    "#He has a",
    "#He got all",
    "#Everyone is happy and I can",
    "#The new movie that got Oscar this year",
    "#In the far far distance from our galaxy,",
    "#Peace is the only way",
]

input_sentences = [inp * scale_input_factor for inp in input_sentences]
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

torch.cuda.empty_cache()
gc.collect()

if batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(batch_size / len(input_sentences))

inputs = input_sentences[:batch_size]

print(f"*** Inference engine used - {inference_engine}")
if inference_engine == "hf":
    # with init_empty_weights():
    # with fast_init(device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_8bit=load_in_8bit,
        device_map="auto",
    )
    model.eval()
    if not load_in_8bit:
        model.to(device)
    generate_fn = hf_generate
elif inference_engine == "hf_pipeline":
    model, tokenizer = init_pipeline_model(model_name, device)
    generate_fn = hfp_generate
elif inference_engine == "vllm":
    model = LLM(model=model_name, dtype="float16")
    generate_fn = vllm_gen_generate
elif inference_engine == "tgi":
    generate_fn = tgt_generate
    model = TG_Pipeline(
        model_type="flash",
        pretrained_model=model_name,
        tokenizer=model_name,
        device="cuda:0",
        dtype="float16",
        config_args=[],
        fast_init=False,
        is_flash=not pargs.no_flash,
        quantize=pargs.quantize,
    )
elif inference_engine == "gptq":
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    max_memory = dict()
    if torch.cuda.is_available():
        max_memory.update({i: f"40GIB" for i in range(torch.cuda.device_count())})
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        quantize_config=BaseQuantizeConfig(),
        max_memory=max_memory,
        use_safetensors=True,
    )
    model.eval()
    if not load_in_8bit:
        model.to(device)
    generate_fn = hf_generate
    tokenizer = None
else:
    pass

try:
    print("*** Running warmup generate")
    t_generate_start = time.time()
    # benchmark
    t0 = time.time()
    pairs = generate_fn(model, tokenizer, input_sentences, **generate_kwargs)
    total_new_tokens = sum(total_new_tokens for (_, _, total_new_tokens) in pairs)
    t_generate_span = time.time() - t_generate_start

    for i, o, _ in pairs:
        print(f"{'-'*60}\nin={i}\nout={o}\n")
    print(f"{'-'*60}\n Time to generate: {t_generate_span}")

    torch.cuda.synchronize()

    print("*** Running benchmark")

    # deepspeed.runtime.utils.see_memory_usage("end-of-run", force=True)
    # torch.cuda.empty_cache()
    # gc.collect()
    total_new_tokens_generated = total_new_tokens
    for i in range(cycles):
        cycle_start = time.time()
        outputs = generate_fn(
            model=model, tokenizer=tokenizer, inputs=inputs, **generate_kwargs
        )
        total_new_tokens_generated += sum(
            total_new_tokens for (_, _, total_new_tokens) in outputs
        )
        cycle_time = time.time() - cycle_start
        print(f"cycle time:{cycle_time}")
    torch.cuda.synchronize()
except torch.cuda.OutOfMemoryError as e:
    print(f"Out of memory during generation")
    cycles_generation_time = -1
    total_new_tokens_generated = -1
    t_generate_span = -1

cycles_generation_time = time.time() - t0
throughput_per_token = (cycles_generation_time) / (total_new_tokens_generated)
throughput = (total_new_tokens_generated) / (cycles_generation_time)

print(
    f"""
*** Performance stats:
Throughput per token including tokenize: {throughput_per_token*1000:.2f} msecs
Throughput (tokens/sec):{throughput}
Tokenize and generate {total_new_tokens_generated} (bs={batch_size}) tokens: {cycles_generation_time:.3f} secs
"""
)

wb.log(
    {
        "throughput_per_token": f"{throughput_per_token*1000:.2f} msecs",
        "total_new_tokens_generated": total_new_tokens_generated,
        "batch_size": batch_size,
        "throughput": f"{throughput} tokens/sec",
        "total_generation_time": t_generate_span,
    }
)
