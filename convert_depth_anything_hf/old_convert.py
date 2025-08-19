import argparse
import os
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help="model and tokenizer path",
                        default='./',
                        )
    return parser.parse_args()


def convert_from_pretrained(model_path):
    model_path = model_path.replace('& ', '').replace("\\", '/').replace("'", '') # Convert Windows Path
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(
        pretrained_model_name_or_path=model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.float32)  #float32 for best compatibility
    # Convert to .safetensors format
    model.save_pretrained(model_path, safe_serialization=True)


if __name__ == '__main__':
    # Usage: model path cannot have "-" use "_" instead!!!
    # 1. Put model (pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt.index or flax_model.msgpack) into a folder ModelName/
    # 2. Run the following:
    # python convert.py --model_path path_to_ModelName

    args = parse_arguments()

    print(f"covert {args.model_path} into HuggingFace safetensor")
    convert_from_pretrained(args.model_path)
