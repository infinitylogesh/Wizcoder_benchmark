from transformers import pipeline,AutoConfig,AutoTokenizer,AutoModelForCausalLM
import torch

def init_pipeline_model(model_name,device,dtype=torch.float16,load_in_8bit=False):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        load_in_8bit=load_in_8bit,
        device_map="auto"
    )
    
    if not load_in_8bit:
        model.to(device)
        
    model = model.eval()

    pipe = pipeline(
        'text-generation',
        model=model,
        config=config,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=dtype,
    )
    
    return pipe,tokenizer

def generate(model,tokenizer,inputs,**generate_kwargs):
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt",add_special_tokens=False, padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    
    generate_kwargs["return_tensors"]=True
    #generate_kwargs["return_text"]=True
    outputs = model(inputs,**generate_kwargs)
    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [len(x[0]['generated_token_ids']) for x in outputs]
    output_tokens = [x[0]['generated_token_ids'] for x in outputs] 
    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    return zip(inputs, output_texts, total_new_tokens)
