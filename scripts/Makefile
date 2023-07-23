install-tgi:
	git clone https://github.com/huggingface/text-generation-inference.git
	cd text-generation-inference && git fetch && git checkout v0.8.2
	make install && make install-benchmark && cd ..

install-vllm:
	git clone https://github.com/vllm-project/vllm.git
	cd vllm && git fetch && git checkout 6fc2a38b110f9ba6037b31ee016f20df32426877
	pip install -e . && cd ..

