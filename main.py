import time,math
from hf_vanilla import generate as hf_generate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from fast_init import fast_init
from pipeline_gen import generate as tgt_generate
from pipeline import TG_Pipeline
from vllm_gen import vllm_gen_generate
from vllm import LLM
import gc
#import deepspeed
#from accelerate import init_empty_weights

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--num_tokens",type=int,default=128)
parser.add_argument("--batch_size",type=int,default=16)
parser.add_argument("--load_in_8bit",action="store_true")
parser.add_argument("--model_name",type=str,default="WizardLM/WizardCoder-15B-V1.0")
parser.add_argument("--inference_engine",type=str,default=None)

pargs = parser.parse_args()

num_tokens = pargs.num_tokens
batch_size = pargs.batch_size
model_name = pargs.model_name
load_in_8bit = pargs.load_in_8bit
inference_engine = pargs.inference_engine
cycles = 3

wb = wandb.init(project="wizcoder_benchmark",config=pargs.__dict__)

t_start = time.time()
#model_name = "bigcode/starcoder"
device = torch.device("cuda:0")
generate_kwargs = dict(min_new_tokens=num_tokens,
                       max_new_tokens=num_tokens, 
                       do_sample=False,
                       use_cache=True,
                       temperature=0)

warmup_sentences = [
    "def",
]*batch_size

input_sentences = [
    "def test_"*100,
    "def api_"*100,
    "#DeepSpeed is a machine learning framework"*100,
    "#He is working on"*100,
    "#He has a"*100,
    "#He got all"*100,
    "#Everyone is happy and I can"*100,
    "#The new movie that got Oscar this year"*100,
    "#In the far far distance from our galaxy,"*100,
    "#Peace is the only way"*100,
 ]

torch.cuda.empty_cache()
gc.collect()

if batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(batch_size / len(input_sentences))

inputs = input_sentences[: batch_size]

tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")

print(f"*** Inference engine used - {inference_engine}")
if inference_engine=="hf":
    #with init_empty_weights():
    #with fast_init(device):
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,load_in_8bit=load_in_8bit,device_map="auto")
    model.eval()
    if not load_in_8bit:
        model.to(device)
    generate_fn = hf_generate
elif inference_engine=="vllm":
    model = LLM(model=model_name,dtype="float16")
    generate_fn = vllm_gen_generate
elif inference_engine=="tgi":
    generate_fn = tgt_generate
    model = TG_Pipeline(model_type="flash",
                        pretrained_model=model_name,
                        tokenizer=model_name,
                        device="cuda:0",
                        dtype="float16",
                        config_args=[],
                        fast_init=False)
    tokenizer = None
else:
    pass


print("*** Running warmup generate")
t_generate_start = time.time()
# benchmark
t0 = time.time()
pairs = generate_fn(model,tokenizer,warmup_sentences,**generate_kwargs)
#pairs = text_gen_generate(model_name,inputs,**generate_kwargs)
#outputs= vllm_gen_generate(model,tokenizer,inputs,**generate_kwargs)
#pairs = [(text,out.outputs[0].text,None) for text,out in zip(inputs,outputs)]
total_new_tokens = sum(total_new_tokens for (_,_,total_new_tokens) in pairs)
t_generate_span = time.time() - t_generate_start
for i, o, _ in pairs:
    print(f"{'-'*60}\nin={i}\nout={o}\n")
print(f"{'-'*60}\n Time to generate: {t_generate_span}")
torch.cuda.synchronize()

print("*** Running benchmark")


#deepspeed.runtime.utils.see_memory_usage("end-of-run", force=True)
#torch.cuda.empty_cache()
#gc.collect()
total_new_tokens_generated = total_new_tokens
for i in range(cycles):
    cycle_start = time.time() 
    outputs= generate_fn(model=model,tokenizer=tokenizer,inputs=inputs,**generate_kwargs)
    total_new_tokens_generated += sum(total_new_tokens for (_,_,total_new_tokens) in outputs)
    cycle_time = time.time() - cycle_start 
    print(f"cycle time:{cycle_time}")
    
torch.cuda.synchronize()
cycles_generation_time = time.time() - t0
throughput_per_token = (cycles_generation_time)/ (total_new_tokens_generated)
throughput = (total_new_tokens_generated) / (cycles_generation_time)
print(
    f"""
*** Performance stats:
Throughput per token including tokenize: {throughput_per_token*1000:.2f} msecs
Throughput (tokens/sec):{throughput}
Tokenize and generate {total_new_tokens_generated} (bs={batch_size}) tokens: {t_generate_span:.3f} secs
"""
)

wb.log({"throughput_per_token":f"{throughput_per_token*1000:.2f} msecs",
       "total_new_tokens_generated":total_new_tokens_generated,
       "batch_size":batch_size,
       "throughput": f"{throughput} tokens/sec",
       "total_generation_time":t_generate_span})

    
    