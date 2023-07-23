# WizardCorder/Starcoder Benchmark



## Installation steps:



## Methodology


## Results:

- **Flash Attention** (Implemented from Text-generation-inference) Performs the best in various setting. However, with long sequences(especially with long input sequences), It seems to result into OOM

- **Paged Attention** (via vLLM) - Performs second best in our benchmark runs and It is better at handling long sequences even in the settings where Flash attention fails , vLLM completes the generation without OOM

- **HF Generate** (baseline) - Huggingface's venilla `AutoModelForCausalLM` taken as a baseline.

- **HF Pipeline** - Huggingface's Pipeline for text-generation performed the worst of all (Results are to be added). 

### Results with short sequence inputs

<p float="left">
  <img src="assets/max_256_shot_input.png" width="350" />
  <img src="assets/max_512_short_input.png" width="350" />  
</p>

### Results with long sequence inputs

- With Batch size of 64 , HF baseline throwed OOM. Flash attention performed better than Paged attention.
- With Batch size of 128, Both HF and Flash attention throwed OOM. Paged attention completed the generations.

<p float="left">
  <img src="assets/max_256_long_input.png" width="350" />
  <img src="assets/max_512_long_input.png" width="350" />  
</p>

### TODO:
For further improvement in throughput, 

- [ ] Performance comparison Quantized model ([GPTQ](TheBloke/WizardCoder-15B-1.0-GPTQ))
- [ ] Flash Attention + Paged Attention ( Using latest Text-generation-inference )
- [ ] [Continous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [ ] Other optimizations listed [here](https://github.com/huggingface/text-generation-inference/issues/376).

## References & Credits :

- Flash attention implementation was used from [Text-generation-inference](https://github.com/huggingface/text-generation-inference/blob/v0.8.2/server/text_generation_server/models/flash_santacoder.py). Adapted the TGI wrapper from Bigcode's [bigcode-inference-benchmark](https://github.com/bigcode-project/bigcode-inference-benchmark)
- Paged Attention from [vLLM](https://github.com/vllm-project/vllm/tree/main)