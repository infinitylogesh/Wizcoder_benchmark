from vllm import LLM, SamplingParams
import inspect

def vllm_gen_generate(model,tokenizer,inputs,**generate_kwargs):
    #input_tokens = tokenizer.batch_encode_plus(inputs,add_special_tokens=False, padding=True)
    generate_kwargs["max_tokens"] = generate_kwargs['max_new_tokens']
    generate_kwargs["ignore_eos"] = True
    generate_kwargs = {k:v for k,v in generate_kwargs.items() if k in inspect.signature(SamplingParams.__init__).parameters.keys()} 
    print(generate_kwargs)
    for prompt in inputs:
        sampling_params = SamplingParams(**generate_kwargs)
        # FIXME(woosuk): Do not use internal method.
        model._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )
    #sampling_params = SamplingParams(**generate_kwargs)
    outputs = model._run_engine(use_tqdm=True)
    pairs = [(text,out.outputs[0].text,len(out.outputs[0].token_ids)) for text,out in zip(inputs,outputs)] 
    return pairs