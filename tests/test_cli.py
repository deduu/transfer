import sys
import pytest
from unittest.mock import patch, MagicMock

from transfer.cli import main


def _parse(argv):
    """Run main()'s argparse with the given argv and return the parsed args.

    We patch sys.argv and intercept the command handlers so we only test
    argument parsing, not execution.
    """
    import argparse
    with patch.object(sys, "argv", ["transfer"] + argv):
        from transfer.cli import main as _main
        # Re-import to get a fresh parser
        import importlib
        import transfer.cli as cli_module
        importlib.reload(cli_module)

        # Capture args by patching the command functions
        captured = {}

        def fake_train(args):
            captured["args"] = args

        def fake_infer(args):
            captured["args"] = args

        with patch.object(cli_module, "train_command", fake_train), \
             patch.object(cli_module, "inference_command", fake_infer):
            cli_module.main()

        return captured.get("args")


class TestTrainCommand:
    def test_train_required_args(self):
        args = _parse([
            "train",
            "--task", "sft",
            "--model_name", "gpt2",
            "--dataset_path", "data.json",
            "--output_dir", "./out",
        ])
        assert args.task == "sft"
        assert args.model_name == "gpt2"
        assert args.dataset_path == "data.json"
        assert args.output_dir == "./out"

    def test_train_missing_task_exits(self):
        with pytest.raises(SystemExit):
            _parse([
                "train",
                "--model_name", "gpt2",
                "--dataset_path", "data.json",
                "--output_dir", "./out",
            ])

    def test_train_invalid_task_exits(self):
        with pytest.raises(SystemExit):
            _parse([
                "train",
                "--task", "rlhf",
                "--model_name", "gpt2",
                "--dataset_path", "data.json",
                "--output_dir", "./out",
            ])

    def test_train_default_values(self):
        args = _parse([
            "train",
            "--task", "sft",
            "--model_name", "gpt2",
            "--dataset_path", "data.json",
            "--output_dir", "./out",
        ])
        assert args.batch_size == 4
        assert args.learning_rate == 2e-4
        assert args.num_epochs == 3
        assert args.max_length == 512

    def test_train_lora_flag(self):
        args = _parse([
            "train",
            "--task", "sft",
            "--model_name", "gpt2",
            "--dataset_path", "data.json",
            "--output_dir", "./out",
            "--use_lora",
        ])
        assert args.use_lora is True

    def test_train_sft_specific_args(self):
        args = _parse([
            "train",
            "--task", "sft",
            "--model_name", "gpt2",
            "--dataset_path", "data.json",
            "--output_dir", "./out",
            "--messages_column", "conversations",
            "--train_on_completions_only",
            "--system_prompt", "You are helpful.",
        ])
        assert args.messages_column == "conversations"
        assert args.train_on_completions_only is True
        assert args.system_prompt == "You are helpful."

    def test_train_loop_args(self):
        args = _parse([
            "train",
            "--task", "sft",
            "--model_name", "gpt2",
            "--dataset_path", "data.json",
            "--output_dir", "./out",
            "--gradient_accumulation_steps", "4",
            "--warmup_steps", "100",
            "--scheduler_type", "linear",
            "--max_grad_norm", "0.5",
            "--save_steps", "500",
            "--logging_steps", "20",
        ])
        assert args.gradient_accumulation_steps == 4
        assert args.warmup_steps == 100
        assert args.scheduler_type == "linear"
        assert args.max_grad_norm == 0.5
        assert args.save_steps == 500
        assert args.logging_steps == 20


class TestInferCommand:
    def test_infer_required_args(self):
        args = _parse([
            "infer",
            "--model_name", "gpt2",
            "--prompt", "Hello world",
        ])
        assert args.model_name == "gpt2"
        assert args.prompt == "Hello world"

    def test_infer_prompt_required(self):
        with pytest.raises(SystemExit):
            _parse([
                "infer",
                "--model_name", "gpt2",
            ])

    def test_infer_mutual_exclusion(self):
        with pytest.raises(SystemExit):
            _parse([
                "infer",
                "--model_name", "gpt2",
                "--prompt", "Hello",
                "--prompt_file", "prompt.txt",
            ])

    def test_infer_default_values(self):
        args = _parse([
            "infer",
            "--model_name", "gpt2",
            "--prompt", "Hello",
        ])
        assert args.max_new_tokens == 256
        assert args.temperature == 0.7
        assert args.top_p == 0.9


class TestNoCommand:
    def test_no_command(self):
        """No subcommand should result in command=None (prints help)."""
        args = _parse([])
        # When no command is given, args is None because no handler was called
        assert args is None
