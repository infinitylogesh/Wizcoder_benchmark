from vllm import LLM, SamplingParams
import inspect

def vllm_gen_generate(model,tokenizer,texts,**generate_kwargs):
    input_tokens = tokenizer.batch_encode_plus(texts,add_special_tokens=False, padding=True)
    generate_kwargs["max_tokens"] = len(input_tokens)+generate_kwargs['max_new_tokens']
    generate_kwargs["ignore_eos"] = True
    generate_kwargs = {k:v for k,v in generate_kwargs.items() if k in inspect.signature(SamplingParams.__init__).parameters.keys()} 
    print(generate_kwargs)
    sampling_params = SamplingParams(**generate_kwargs)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=512)
    outputs = model.generate(texts,sampling_params)
    pairs = [(text,out.outputs[0].text,len(out.outputs[0].token_ids)) for text,out in zip(texts,outputs)]
    # prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    # ]
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)   
    # outputs = model.generate(prompts, sampling_params)
    # pairs = [(text,out.outputs[0].text,len(out.outputs[0].token_ids)) for text,out in zip(prompts,outputs)]
    return pairs