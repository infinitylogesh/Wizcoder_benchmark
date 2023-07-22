
custom_generate = True 
do_prefill = True
use_cache = True
breakdown_latency = True
key_length_step = 1
ignore_oom = True
pad_generated_tokens = 0 

def generate(model,tokenizer,inputs,**generate_kwargs):

    generated_texts, metrics = model(
                    inputs,
                    generate_kwargs["max_new_tokens"],
                    custom_generate=custom_generate,
                    use_cache=use_cache,
                    do_prefill=do_prefill,
                    breakdown_latency=breakdown_latency,
                    key_length_step=key_length_step,
                    ignore_oom=ignore_oom,
                    pad_generated_tokens=pad_generated_tokens,
                )

    return zip(inputs,generated_texts,[metrics['Tokens generated (sample)']]*metrics['Batch size'])