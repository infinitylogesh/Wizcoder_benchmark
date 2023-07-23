flash_att_commit := 3a9bfd076f98746c73362328958dbc68d145fbec


install-tgi:
	git clone https://github.com/huggingface/text-generation-inference.git
	cd text-generation-inference && git fetch && git checkout v0.8.2
	cd text-generation-inference && make install && make install-benchmark && cd ..
	
install-flash-attn:
	cd text-generation-inference/server && make Makefile-flash-att install-flash-attention && cd ..

install-vllm:
	git clone https://github.com/vllm-project/vllm.git
	cd vllm && git fetch && git checkout 6fc2a38b110f9ba6037b31ee016f20df32426877
	cd vllm && pip install -e . && cd ..

