import os
import pytest
from transfer.config import BaseConfig, SFTConfig, DPOConfig


# ── SFT Config ──────────────────────────────────────────────────────────────

class TestSFTConfig:
    def test_sft_valid_defaults(self, tmp_path):
        cfg = SFTConfig(output_dir=str(tmp_path / "sft"))
        assert cfg.batch_size == 1
        assert cfg.learning_rate == 5e-5
        assert cfg.num_epochs == 30

    def test_sft_custom_params(self, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=5,
            max_length=128,
        )
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 1e-4
        assert cfg.num_epochs == 5
        assert cfg.max_length == 128

    def test_sft_batch_size_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="batch_size"):
            SFTConfig(output_dir=str(tmp_path / "sft"), batch_size=0)

    def test_sft_learning_rate_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="learning_rate"):
            SFTConfig(output_dir=str(tmp_path / "sft"), learning_rate=0)

    def test_sft_learning_rate_negative_raises(self, tmp_path):
        with pytest.raises(ValueError, match="learning_rate"):
            SFTConfig(output_dir=str(tmp_path / "sft"), learning_rate=-0.01)

    def test_sft_invalid_quantization_type(self, tmp_path):
        with pytest.raises(ValueError, match="quantization_type"):
            SFTConfig(output_dir=str(tmp_path / "sft"), quantization_type="int8")

    @pytest.mark.parametrize("qtype", ["nf4", "fp4"])
    def test_sft_valid_quantization_types(self, tmp_path, qtype):
        cfg = SFTConfig(output_dir=str(tmp_path / "sft"), quantization_type=qtype)
        assert cfg.quantization_type == qtype

    def test_sft_invalid_compute_dtype(self, tmp_path):
        with pytest.raises(ValueError, match="quantization_compute_dtype"):
            SFTConfig(output_dir=str(tmp_path / "sft"), quantization_compute_dtype="float8")

    @pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
    def test_sft_valid_compute_dtypes(self, tmp_path, dtype):
        cfg = SFTConfig(output_dir=str(tmp_path / "sft"), quantization_compute_dtype=dtype)
        assert cfg.quantization_compute_dtype == dtype

    def test_sft_grad_accum_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            SFTConfig(output_dir=str(tmp_path / "sft"), gradient_accumulation_steps=0)

    def test_sft_max_grad_norm_negative_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_grad_norm"):
            SFTConfig(output_dir=str(tmp_path / "sft"), max_grad_norm=-1)

    def test_sft_max_grad_norm_zero_ok(self, tmp_path):
        cfg = SFTConfig(output_dir=str(tmp_path / "sft"), max_grad_norm=0.0)
        assert cfg.max_grad_norm == 0.0

    def test_sft_output_dir_nesting(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Use default relative output_dir so os.path.join nests properly
        cfg = SFTConfig()
        # output_dir should be nested under BaseConfig.output_dir ("./finetuned_model")
        assert "finetuned_model" in cfg.output_dir
        assert "sft_finetuned_model" in cfg.output_dir

    def test_sft_specific_defaults(self, tmp_path):
        cfg = SFTConfig(output_dir=str(tmp_path / "sft"))
        assert cfg.messages_column is None
        assert cfg.train_on_completions_only is False
        assert cfg.system_prompt is None


# ── DPO Config ──────────────────────────────────────────────────────────────

class TestDPOConfig:
    def test_dpo_valid_defaults(self, tmp_path):
        cfg = DPOConfig(output_dir=str(tmp_path / "dpo"))
        assert cfg.beta == 0.1
        assert cfg.learning_rate == 1e-6

    def test_dpo_beta_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="beta"):
            DPOConfig(output_dir=str(tmp_path / "dpo"), beta=0)

    def test_dpo_beta_negative_raises(self, tmp_path):
        with pytest.raises(ValueError, match="beta"):
            DPOConfig(output_dir=str(tmp_path / "dpo"), beta=-0.1)

    def test_dpo_batch_size_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="batch_size"):
            DPOConfig(output_dir=str(tmp_path / "dpo"), batch_size=0)

    def test_dpo_grad_accum_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            DPOConfig(output_dir=str(tmp_path / "dpo"), gradient_accumulation_steps=0)

    def test_dpo_specific_defaults(self, tmp_path):
        cfg = DPOConfig(output_dir=str(tmp_path / "dpo"))
        assert cfg.learning_rate == 1e-6
        assert cfg.beta == 0.1
        assert cfg.max_length == 512
        assert cfg.chosen_column == "chosen"
        assert cfg.rejected_column == "rejected"
