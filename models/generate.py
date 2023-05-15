import json
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import torch
import yaml
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

sys.path.insert(0, Path(__file__).parent.parent.absolute().as_posix())
from models.dataset import DialogueDataModule, SpecialVocab

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("generate")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_LENGTH = int(1000)  # Hardcoded max length to avoid infinite loop

def set_default_args(arguments):
    if not arguments.control_tokens:
        arguments.control_tokens = []
    elif arguments.control_tokens[0] == "none":
        arguments.control_tokens = []
    elif arguments.control_tokens[0] == "all":
        arguments.control_tokens = ["<no-first-person>", "<high-prec>", "<entailed>"]

    arguments.do_generate = True
    arguments.predict_dataset_path = arguments.dataset_path

    arguments.do_train = False
    arguments.do_eval = False
    arguments.do_test = False
    arguments.ctrl = False
    arguments.max_negative_samples = 0

    arguments.pad_to_multiple_of = None

    hparams_path = Path(arguments.model_name_or_path).parent / "hparams.yaml"
    if hparams_path.exists():
        logger.info(
            "`hparams.yaml` found from which parameter values (max_history, pad_to_multiple_of) will be loaded"
        )

        with hparams_path.open("r") as hparams_file:
            train_hparams = yaml.safe_load(hparams_file)

        arguments.pad_to_multiple_of = train_hparams.get("pad_to_multiple_of", None)
        arguments.max_history = arguments.max_history or train_hparams.get("max_history", None)
        arguments.ctrl = train_hparams.get("ctrl", False)


def get_output_name(arguments) -> str:
    name = "generated"
    if arguments.dataset_path:
        name += f"_{Path(arguments.dataset_path).stem}"

    if arguments.num_return_sequences > 1:
        name += f"_n{arguments.num_return_sequences}"

    name += f"_maxHist{arguments.max_history}_maxLen{arguments.max_length}"

    if arguments.temperature != 1.0:
        name += f"_temp{arguments.temperature}"

    if arguments.repetition_penalty != 1.0:
        name += f"_repPen{arguments.repetition_penalty}"

    if arguments.do_sample:
        if arguments.top_k > 0:
            name += f"_k{arguments.top_k}"
        if arguments.top_p > 0:
            name += f"_p{arguments.top_p}"
    else:
        name += "_greedy"

    if arguments.ctrl and arguments.control_tokens:
        ctrl_tokens = ",".join([token[1:-1] for token in arguments.control_tokens])
        name += f"_{ctrl_tokens}"

    return name

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset_path", type=str, default=None, help="Path or url of the Json dataset.")
    arg_parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to a trained model")
    arg_parser.add_argument("--max_seq_length", type=int, default=512)
    arg_parser.add_argument("--output", type=str, default=None, help="Path of the output directory to save the responses")
    arg_parser.add_argument(
        "--max_history",
        type=int,
        default=None,
        help="Number of previous exchanges to keep in history",
    )
    arg_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    arg_parser.add_argument("--max_length", type=int, default=100)
    arg_parser.add_argument("--min_length", type=int, default=2)
    arg_parser.add_argument(
        "--temperature", type=float, default=1.0, help="value used to module the next token probabilities"
    )
    arg_parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Whether or not to use sampling ; use greedy decoding otherwise.",
    )
    arg_parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    arg_parser.add_argument("--top_k", type=int, default=0)
    arg_parser.add_argument("--top_p", type=float, default=0)
    arg_parser.add_argument("--num_return_sequences", type=int, default=1)
    arg_parser.add_argument(
        "--exclude_knowledge",
        action="store_true",
        default=False,
        help="Whether to exclude knowledge from input sequences",
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    arg_parser.add_argument("--num_workers", default=10, type=int, help="kwarg passed to DataLoader")
    arg_parser.add_argument(
        "--control_tokens",
        nargs="*",
        default=("<entailed>",),
        help="Prepend control tokens to the sequence for controlled generation "
        "(works only when model is trained with `--ctrl`). List of control tokens are: "
        "<entailed>, <non-entailed>, <first-person>, <no-first-person>, <high-prec>, <med-prec>, <low-prec>. "
        "To use all of them, simply pass `--control_tokens all` and for none, pass `--control_tokens none`.",
    )

    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    set_default_args(args)
    logger.info(f"Arguments: {pformat(args)}")

    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        return_dict=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path=args.model_name_or_path, extra_ids=0)
    except ValueError:
        logger.warning(
            "Creating tokenizer failed, trying again without extra_ids (used only for T5). "
            "In this setting, the model may generate reserved tokens (<extra_id_%%>)."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path=args.model_name_or_path)

    if args.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path=args.model_name_or_path, config=args.config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path=args.model_name_or_path, config=args.config)

    special_vocab = SpecialVocab(tokenizer=tokenizer, ctrl=args.ctrl, initialized=True)

    model.to(device=args.device)

    if args.max_length < 0 and tokenizer.model_max_length > 0:
        max_length = tokenizer.model_max_length
    elif 0 < tokenizer.model_max_length < args.max_length:
        max_length = tokenizer.model_max_length  # No generation bigger than model size
    elif args.max_length < 0:
        max_length = MAX_LENGTH  # avoid infinite loop

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    data_module = DialogueDataModule(special_vocab=special_vocab, args=args, is_encoder_decoder=args.is_encoder_decoder)
    data_module.setup("fit")

    logger.info(f"Test dataset size: {len(data_module.datasets['generate'])}")

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(model_name_or_path=args.model_name_or_path) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{get_output_name(args)}.jsonl"
    logger.info(f"Results will be saved in `{output_file}`")

    example_idx = 0
    with output_file.open("w", encoding="utf-8") as writer:
        predict_dataloader = data_module.predict_dataloader()
        for batch in tqdm(predict_dataloader, total=len(predict_dataloader)):
            model.eval()
            batch = {k: t.to(args.device) for k, t in batch.items()}
            input_ids = batch["input_ids"]

            if "token_type_ids" in batch:
                gen_kwargs = {"token_type_ids": batch["token_type_ids"]}
            else:
                gen_kwargs = {}

            input_lengths = (input_ids != tokenizer.pad_token_id).int().sum(-1)

            # responses: (batch_size * num_return_sequences, sequence_length)
            responses = getattr(model, "module", model).generate(
                input_ids,
                decoder_start_token_id=special_vocab.wizard_token_id,
                do_sample=args.do_sample,
                max_length=(0 if args.is_encoder_decoder else input_ids.shape[-1]) + max_length,
                min_length=args.min_length,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                num_return_sequences=args.num_return_sequences,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **gen_kwargs,
            )

            batch_size = input_ids.shape[0]

            # responses: (batch_size, num_return_sequences, sequence_length)
            responses = responses.reshape(batch_size, args.num_return_sequences, -1)
            responses = responses.cpu().numpy()

            for b in range(batch_size):
                example = data_module.datasets["generate"][example_idx]
                out = {
                    "dialog_idx": example["dialog_idx"],
                    "response": example["response"],
                    "history": example["history"],
                    "knowledge": example["knowledge"],
                }
                if "original_response" in example:
                        out["original_response"] = example["original_response"]

                if "BEGIN" in example:
                    out["BEGIN"] = example["BEGIN"]

                if "VRM" in example:
                    out["VRM"] = example["VRM"]

                generated_responses = [
                    tokenizer.decode(
                        responses[b, i] if config.is_encoder_decoder else responses[b, i, input_lengths[b] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    for i in range(args.num_return_sequences)
                ]

                out["generated_response"] = [resp for resp in generated_responses if resp]

                if not out["generated_response"]:
                    logger.warning(f"Empty generated response at {example_idx}: {out}")

                writer.write(json.dumps(out) + "\n")

                example_idx += 1


if __name__ == "__main__":
    main()