from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

def generate(model,tokenizer,inputs,**generate_kwargs):
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt",add_special_tokens=False, padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    
    outputs = model.generate(input_ids=input_tokens['input_ids'], **generate_kwargs)
    
    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)
