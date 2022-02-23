import argparse
import torch
from catbird.tokenizers import build_tokenizer
from ignite.utils import manual_seed, setup_logger
from catbird.core import Config, mkdir_or_exist
from pathlib import Path
from ignite.handlers import Checkpoint
from catbird.models import build_generator
import ignite.distributed as idist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paraphrase a file using pretrained model"
    )

    parser.add_argument("text", help="input text to paraphrase")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--work-dir", help="dir to save logs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("-o", "--output", help="output file")
    parser.add_argument("-m", "--model", help="model checkpoint file")
    parser.add_argument("--beam-size", default=5, type=int, help="beam size used")
    parser.add_argument(
        "--max-input-length", default=50, type=int, help="maximum input length"
    )
    parser.add_argument(
        "--max-output-length", default=50, type=int, help="maximum output length"
    )
    parser.add_argument(
        "--length-normalization",
        default=0.6,
        type=float,
        help="length normalization factor",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    manual_seed(args.seed)

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = Path("./work_dirs", Path(args.config).stem)

    mkdir_or_exist(Path(cfg.work_dir).resolve())

    logger = setup_logger(name="Inference")

    tokenizer = build_tokenizer(cfg)
    cfg.embedding_length = len(tokenizer)
    cfg.pad_token_id = tokenizer.pad_token_id

    model = build_generator(cfg)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.info(f"Model's # of parameters: {num_parameters}")

    checkpoint = torch.load(args.checkpoint)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    input_text = [args.text]
    
    input_txt_tokenized = tokenizer(
            input_text, max_length=args.max_input_length, padding="max_length", truncation=True
        )
    
    batch = {
            k: torch.tensor(v).to(idist.device()) for (k, v) in input_txt_tokenized.items()
        }
    src_ids = batch["input_ids"]
    src_attention_mask = batch["attention_mask"]
    
    with torch.no_grad():
        y_preds = model.generate(src_ids, attention_mask=src_attention_mask)
    
    def ids_to_clean_text(generated_ids):
        gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return list(map(str.strip, gen_text))
    
    preds = ids_to_clean_text(y_preds)
    preds = [_preds.split() for _preds in preds]
    
    logger.info(f'\nPreds: {" ".join(preds[0])} \
                \nTarget: {input_text[0]}')
