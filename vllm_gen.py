from vllm import LLM, SamplingParams
import inspect

def vllm_gen_generate(model,tokenizer,texts,**generate_kwargs):
    input_tokens = tokenizer.batch_encode_plus(texts,add_special_tokens=False, padding=True)
    generate_kwargs["max_tokens"] = len(input_tokens)+generate_kwargs['max_new_tokens']
    generate_kwargs = {k:v for k,v in generate_kwargs.items() if k in inspect.signature(SamplingParams.__init__).parameters.keys()} 
    sampling_params = SamplingParams(**generate_kwargs)
    outputs = model.generate(prompt_token_ids=input_tokens['input_ids'], sampling_params=sampling_params)
    pairs = [(text,out.outputs[0].text,len(out.outputs[0].token_ids)) for text,out in zip(texts,outputs)]
    return pairs