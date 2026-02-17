import os
import csv
import math
import pytest
import torch
from unittest.mock import patch, MagicMock, call

from transfer.config import SFTConfig, DPOConfig
from transfer.strategies.sft import SFTStrategy
from transfer.strategies.dpo import DPOStrategy
from transfer.trainer import Trainer

from tests.conftest import FakeModel


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.pad_token_id = 0
    tok.eos_token = "</s>"
    tok.eos_token_id = 2
    tok.model_max_length = 512
    # save_pretrained is needed by save_model
    tok.save_pretrained = MagicMock()
    return tok


@pytest.fixture
def _patch_load(fake_model):
    """Patch load_model_and_tokenizer to return (FakeModel, mock_tokenizer)."""
    tok = _make_mock_tokenizer()
    with patch("transfer.trainer.load_model_and_tokenizer", return_value=(fake_model, tok)):
        yield fake_model, tok


# ── Strategy creation ────────────────────────────────────────────────────────

class TestStrategyCreation:
    def test_sft_creates_sft_strategy(self, _patch_load, tmp_path):
        cfg = SFTConfig(output_dir=str(tmp_path / "sft"), quantize=False)
        trainer = Trainer(task="sft", config=cfg)
        assert isinstance(trainer.strategy, SFTStrategy)

    def test_dpo_creates_dpo_strategy(self, _patch_load, tmp_path):
        cfg = DPOConfig(output_dir=str(tmp_path / "dpo"), quantize=False)
        # Patch the second call to load_model_and_tokenizer for reference model
        with patch("transfer.trainer.load_model_and_tokenizer",
                    return_value=(FakeModel(), _make_mock_tokenizer())):
            trainer = Trainer(task="dpo", config=cfg)
        assert isinstance(trainer.strategy, DPOStrategy)

    def test_invalid_task_raises(self, _patch_load, tmp_path):
        cfg = SFTConfig(output_dir=str(tmp_path / "sft"), quantize=False)
        with pytest.raises(ValueError, match="not supported"):
            Trainer(task="rlhf", config=cfg)

    def test_pad_token_set(self, tmp_path, fake_model):
        tok = _make_mock_tokenizer()
        tok.pad_token = None
        with patch("transfer.trainer.load_model_and_tokenizer", return_value=(fake_model, tok)):
            cfg = SFTConfig(output_dir=str(tmp_path / "sft"), quantize=False)
            trainer = Trainer(task="sft", config=cfg)
        assert trainer.tokenizer.pad_token == tok.eos_token


# ── Gradient accumulation ────────────────────────────────────────────────────

class TestGradientAccumulation:
    def _run_train(self, trainer, num_batches):
        """Simulate a training loop with the given number of batches."""
        # Create a mock dataset
        from datasets import Dataset
        data = {"input_ids": [[1, 2, 3]] * num_batches, "attention_mask": [[1, 1, 1]] * num_batches}
        trainer.train_dataset = Dataset.from_dict(data)

        # Mock strategy methods
        trainer.strategy.preprocess_data = MagicMock(return_value=trainer.train_dataset)
        trainer.strategy.get_data_collator = MagicMock(return_value=None)
        trainer.strategy.compute_loss = MagicMock(
            return_value=torch.tensor(1.0, requires_grad=True)
        )

        # Use MagicMock for optimizer.step (plain functions lack __func__
        # which LambdaLR's patch_track_step_called expects)
        step_mock = MagicMock()
        trainer.optimizer.step = step_mock
        trainer.optimizer.zero_grad = MagicMock()

        trainer.train()
        return step_mock.call_count

    def test_grad_accum_step_count(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            quantize=False,
            batch_size=1,
            num_epochs=1,
            gradient_accumulation_steps=2,
            logging_steps=0,
        )
        trainer = Trainer(task="sft", config=cfg)
        steps = self._run_train(trainer, num_batches=4)
        assert steps == 2

    def test_grad_accum_partial_last(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            quantize=False,
            batch_size=1,
            num_epochs=1,
            gradient_accumulation_steps=3,
            logging_steps=0,
        )
        trainer = Trainer(task="sft", config=cfg)
        # 5 batches, accum=3 -> step at batch 3 and batch 5 (last batch) = 2 steps
        steps = self._run_train(trainer, num_batches=5)
        assert steps == 2


# ── Gradient clipping ────────────────────────────────────────────────────────

class TestGradientClipping:
    def _train_one_step(self, trainer, max_grad_norm):
        """Train for one step and check clip_grad_norm_ calls."""
        from datasets import Dataset
        trainer.config.max_grad_norm = max_grad_norm

        data = {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
        trainer.train_dataset = Dataset.from_dict(data)
        trainer.strategy.preprocess_data = MagicMock(return_value=trainer.train_dataset)
        trainer.strategy.get_data_collator = MagicMock(return_value=None)
        trainer.strategy.compute_loss = MagicMock(
            return_value=torch.tensor(1.0, requires_grad=True)
        )
        trainer.optimizer.step = MagicMock()
        trainer.optimizer.zero_grad = MagicMock()

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            trainer.train()
            return mock_clip

    def test_grad_clipping_applied(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
            batch_size=1, num_epochs=1, max_grad_norm=1.0, logging_steps=0,
        )
        trainer = Trainer(task="sft", config=cfg)
        mock_clip = self._train_one_step(trainer, 1.0)
        mock_clip.assert_called()

    def test_grad_clipping_disabled(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
            batch_size=1, num_epochs=1, max_grad_norm=0.0, logging_steps=0,
        )
        trainer = Trainer(task="sft", config=cfg)
        mock_clip = self._train_one_step(trainer, 0.0)
        mock_clip.assert_not_called()


# ── Scheduler ────────────────────────────────────────────────────────────────

class TestScheduler:
    def test_scheduler_steps_match_optimizer(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
            batch_size=1, num_epochs=1, logging_steps=0,
        )
        trainer = Trainer(task="sft", config=cfg)

        from datasets import Dataset
        data = {"input_ids": [[1, 2]] * 3, "attention_mask": [[1, 1]] * 3}
        trainer.train_dataset = Dataset.from_dict(data)
        trainer.strategy.preprocess_data = MagicMock(return_value=trainer.train_dataset)
        trainer.strategy.get_data_collator = MagicMock(return_value=None)
        trainer.strategy.compute_loss = MagicMock(
            return_value=torch.tensor(1.0, requires_grad=True)
        )

        opt_steps = [0]
        sched_steps = [0]

        original_opt_step = trainer.optimizer.step
        def count_opt(*a, **kw):
            opt_steps[0] += 1
        trainer.optimizer.step = count_opt
        trainer.optimizer.zero_grad = MagicMock()

        with patch("transfer.trainer.get_scheduler") as mock_get_sched:
            mock_scheduler = MagicMock()
            def count_sched(*a, **kw):
                sched_steps[0] += 1
            mock_scheduler.step = count_sched
            mock_get_sched.return_value = mock_scheduler

            trainer.train()

        assert opt_steps[0] == sched_steps[0]


# ── Checkpointing ────────────────────────────────────────────────────────────

class TestCheckpointing:
    def _train_with_save_steps(self, trainer, num_batches, save_steps):
        from datasets import Dataset
        trainer.config.save_steps = save_steps

        data = {"input_ids": [[1, 2]] * num_batches, "attention_mask": [[1, 1]] * num_batches}
        trainer.train_dataset = Dataset.from_dict(data)
        trainer.strategy.preprocess_data = MagicMock(return_value=trainer.train_dataset)
        trainer.strategy.get_data_collator = MagicMock(return_value=None)
        trainer.strategy.compute_loss = MagicMock(
            return_value=torch.tensor(1.0, requires_grad=True)
        )
        trainer.optimizer.step = MagicMock()
        trainer.optimizer.zero_grad = MagicMock()

        with patch.object(trainer, "_save_checkpoint") as mock_save:
            trainer.train()
            return mock_save

    def test_checkpoint_at_save_steps(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
            batch_size=1, num_epochs=1, save_steps=2, logging_steps=0,
        )
        trainer = Trainer(task="sft", config=cfg)
        mock_save = self._train_with_save_steps(trainer, num_batches=4, save_steps=2)
        # Steps 2 and 4 -> called twice
        assert mock_save.call_count == 2
        mock_save.assert_any_call(2)
        mock_save.assert_any_call(4)

    def test_no_checkpoint_when_zero(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
            batch_size=1, num_epochs=1, save_steps=0, logging_steps=0,
        )
        trainer = Trainer(task="sft", config=cfg)
        mock_save = self._train_with_save_steps(trainer, num_batches=4, save_steps=0)
        mock_save.assert_not_called()


# ── CSV logging ──────────────────────────────────────────────────────────────

class TestCSVLogging:
    def _train_and_get_csv(self, trainer, num_batches):
        from datasets import Dataset
        data = {"input_ids": [[1, 2]] * num_batches, "attention_mask": [[1, 1]] * num_batches}
        trainer.train_dataset = Dataset.from_dict(data)
        trainer.strategy.preprocess_data = MagicMock(return_value=trainer.train_dataset)
        trainer.strategy.get_data_collator = MagicMock(return_value=None)
        trainer.strategy.compute_loss = MagicMock(
            return_value=torch.tensor(1.0, requires_grad=True)
        )
        trainer.optimizer.step = MagicMock()
        trainer.optimizer.zero_grad = MagicMock()
        trainer.train()

        csv_path = os.path.join(trainer.config.output_dir, "training_log.csv")
        return csv_path

    def test_csv_log_created(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
            batch_size=1, num_epochs=1, logging_steps=1,
        )
        trainer = Trainer(task="sft", config=cfg)
        csv_path = self._train_and_get_csv(trainer, num_batches=2)
        assert os.path.exists(csv_path)

    def test_csv_log_content(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
            batch_size=1, num_epochs=1, logging_steps=1,
        )
        trainer = Trainer(task="sft", config=cfg)
        csv_path = self._train_and_get_csv(trainer, num_batches=2)

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header row
        assert rows[0] == ["step", "epoch", "loss", "learning_rate"]
        # At least one data row
        assert len(rows) > 1


# ── Save checkpoint & model ──────────────────────────────────────────────────

class TestSaveModel:
    def test_save_checkpoint_creates_dir(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
        )
        trainer = Trainer(task="sft", config=cfg)
        trainer._save_checkpoint(10)
        ckpt_dir = os.path.join(cfg.output_dir, "checkpoint-10")
        assert os.path.isdir(ckpt_dir)

    def test_save_model_calls_pretrained(self, _patch_load, tmp_path):
        fake_model, mock_tok = _patch_load
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
        )
        trainer = Trainer(task="sft", config=cfg)
        # Replace model with a mock to track calls
        trainer.model = MagicMock()
        trainer.model.save_pretrained = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.save_pretrained = MagicMock()
        # Disable evaluation comparison
        trainer.config.enable_evaluation = False

        trainer.save_model()
        trainer.model.save_pretrained.assert_called_once()
        trainer.tokenizer.save_pretrained.assert_called_once()


# ── Loss scaling ─────────────────────────────────────────────────────────────

class TestLossScaling:
    def test_loss_scaled_for_accumulation(self, _patch_load, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"), quantize=False,
            batch_size=1, num_epochs=1,
            gradient_accumulation_steps=4,
            logging_steps=0,
        )
        trainer = Trainer(task="sft", config=cfg)

        from datasets import Dataset
        data = {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
        trainer.train_dataset = Dataset.from_dict(data)
        trainer.strategy.preprocess_data = MagicMock(return_value=trainer.train_dataset)
        trainer.strategy.get_data_collator = MagicMock(return_value=None)

        raw_loss = torch.tensor(4.0, requires_grad=True)
        trainer.strategy.compute_loss = MagicMock(return_value=raw_loss)
        trainer.optimizer.step = MagicMock()
        trainer.optimizer.zero_grad = MagicMock()

        # Track what gets backward'd
        backward_values = []
        original_backward = torch.Tensor.backward

        def tracking_backward(self, *args, **kwargs):
            backward_values.append(self.item())
            # Don't actually backward on leaf requiring grad
            # Just record the value

        with patch.object(torch.Tensor, "backward", tracking_backward):
            trainer.train()

        # loss / accum_steps = 4.0 / 4 = 1.0
        assert len(backward_values) == 1
        assert abs(backward_values[0] - 1.0) < 1e-6
