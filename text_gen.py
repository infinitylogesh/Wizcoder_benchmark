from pipeline import TG_Pipeline
from text_generation_server.models.flash_santacoder import FlashSantacoderSharded
from text_generation_server.models.santacoder import SantaCoder
from text_generation_server.pb import generate_pb2
from text_generation_server.models.model import Model
import torch

def text_gen_generate(model_id,texts,**generate_kwargs):
    model:Model = SantaCoder(model_id,dtype=torch.float16)
    #input_tokens = model.tokenizer.batch_encode_plus(inputs, return_tensors="pt",add_special_tokens=False, padding=True)
    batch_pb = generate_pb2.Batch(
            id=0,
            requests=[
                generate_pb2.Request(
                    id=i,
                    inputs=t,
                    truncate=99999,
                    parameters=generate_pb2.NextTokenChooserParameters(temperature=1.0,
                        top_p=1,
                        typical_p=1,
                        do_sample=False,
                        seed=0,
                        repetition_penalty=1.0,
                        watermark=False,
                    ),
                )
                for i, t in enumerate(texts)
            ],
            size=len(texts),
            max_tokens=0,  # Ignored
        )
    batch = model.batch_type.from_pb(batch_pb, model.tokenizer, torch.float16, model.device)
    #model.warmup(batch,max_total_tokens=64)
    generated, batch = model.generate_token(batch)

    return None