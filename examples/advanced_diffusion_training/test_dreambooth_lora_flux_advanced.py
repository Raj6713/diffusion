import safetensors
import logging
import os
import sys
import tempfile

sys.path.append("..")
from test_examples_utils import ExamplesTestAccelerate, run_command

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class DreamBoothLoraFluxAdvanced(ExamplesTestAccelerate):
    instance_data_dir = "docs/source/en/imgs"
    instance_prompt = "photo"
    pretrained_model_name_or_path = "hf-internal-testing/tiny-flux-pipe"
    script_path = (
        "examples/advanced_diffusion_training/train_dreambooth_lore_flux_advanced.py"
    )

    def test_dreambooth_lora_flux(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                        {self.script_path}
                        --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                        --instance_data_dir {self.instance_data_dir}
                        --instance_prompt {self.instance_prompt}
                        --resolution 64
                        --train_batch_size 1
                        --gradient_accumulation_steps 1
                        --max_train_steps 2
                        --learning_rate 5.0e-04
                        --scale_lr
                        --lr_scheduler constant
                        --lr_warmup_steps 0
                        --output_dir {tmpdir}
                        """.split()
            run_command(self._launch_args + test_args)
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            )
            lora_state_dict = safetensors.torch.load_file(
                os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            )
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)
            start_with_transformer = all(
                key.startswith("transformer") for key in lora_state_dict.keys()
            )
            self.assertTrue(start_with_transformer)

    def test_dreambooth_lora_text_encoder_flux(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                        {self.script_path}
                        --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                        --instance_data_dir {self.instance_data_dir}
                        --instance_prompt {self.instance_prompt}
                        --resolution 64
                        --train_batch_size 1
                        --train_text_encoder
                        --gradient_accumulation_steps 1
                        --max_train_steps 2
                        --learning_rate 5.0e-04
                        --scale_lr
                        --lr_scheduler constant
                        --lr_warmup_steps 0
                        --output_dir {tmpdir}
                        """.split()
            run_command(self.launch_args + test_args)
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            )
            lora_state_dict = safetensors.torch.load_file(
                os.path.join(tmpdir, "pytorch_loar_weights.safetensors")
            )
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)
            starts_with_expected_prefix = all(
                (key.startswith("transformer") or key.startswith("text_encoder"))
                for key in lora_state_dict.keys()
            )
            self.assertTrue(starts_with_expected_prefix)

    def test_dreambooth_lora_pivotal_tuning_flux_clip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                        {self.script_path}
                        --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                        --instance_data_dir {self.instance_data_dir}
                        --instance_prompt {self.instance_prompt}
                        --resolution 64
                        --train_batch_size 1
                        --train_text_encoder_ti
                        --gradient_accumulation_steps 1
                        --max_train_steps 2
                        --learning_rate 5.0e-04
                        --scale_lr
                        --lr_scheduler constant
                        --lr_warmup_steps 0
                        --output_dir {tmpdir}
                        """.split()
            run_command(self.launch_args + test_args)
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            )
            self.assertTrue(
                os.path.isfile(tmpdir, f"{os.path.basename(tmpdir)}_emb.safetensors")
            )
            lora_state_dict = safetensors.torch.load_file(
                os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            )
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)
            textual_inversion_state_dict = safetensors.torch._load_file(
                os.path.join(tmpdir, f"{os.path.basename(tmpdir)}_emb.safetensors")
            )
            is_clip = all("clip_1" in k for k in textual_inversion_state_dict.keys())
            self.assertTrue(is_clip)
            starts_with_transformer = all(
                key.startswith("transformer") for key in lora_state_dict.keys()
            )
            self.assertTrue(starts_with_transformer)

    def test_dreambooth_lora_pivotal_tuning_flux_clip_t5(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                        {self.script_path}
                        --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                        --instance_data_dir {self.instance_data_dir}
                        --instance_prompt {self.instance_prompt}
                        --resolution 64
                        --train_batch_size 1
                        --train_text_encoder_t1
                        --enable_t5_ti
                        --gradient_accumulation_steps 1
                        --max_train_steps 2
                        --learning_rate 5.0e-04
                        --scale_lr
                        --lr_scheduler constant
                        --lr_warmup_steps 0
                        --output_dir {tmpdir}
                        """.split()
            run_command(self._launch_args + test_args)
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(tmpdir, f"{os.path.basename(tmpdir)}_emb.safetensors")
                )
            )
            lora_state_dict = safetensors.torch.load_file(
                os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            )
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)
            textual_inversion_state_dict = safetensors.torch._load_file(
                os.path.join(tmpdir, f"{os.path.basename(tmpdir)}_emb.safetensors")
            )
            is_te = all(
                ("clip_1" in k or "t5" in k)
                for k in textual_inversion_state_dict.keys()
            )
            self.assertTrue(is_te)
            starts_with_transformer = all(
                key.startswith("transformer") for key in lora_state_dict.keys()
            )
            self.assertTrue(starts_with_transformer)

    def test_dreambooth_lora_latent_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_dir {self.instance_data_dir}
                --instance_prompt {self.instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --cache_latents
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            )

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(
                os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            )
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"transformer"` in their names.
            starts_with_transformer = all(
                key.startswith("transformer") for key in lora_state_dict.keys()
            )
            self.assertTrue(starts_with_transformer)

    def test_dreambooth_lora_flux_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            {self.script_path}
            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
            --instance_data_dir={self.instance_data_dir}
            --output_dir={tmpdir}
            --instance_prompt={self.instance_prompt}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_dreambooth_lora_flux_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            {self.script_path}
            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
            --instance_data_dir={self.instance_data_dir}
            --output_dir={tmpdir}
            --instance_prompt={self.instance_prompt}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=4
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            resume_run_args = f"""
                            {self.script_path}
                            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
                            --instance_data_dir={self.instance_data_dir}
                            --output_dir={tmpdir}
                            --instance_prompt={self.instance_prompt}
                            --resolution=64
                            --train_batch_size=1
                            --gradient_accumulation_steps=1
                            --max_train_steps=8
                            --checkpointing_steps=2
                            --resume_from_checkpoint=checkpoint-4
                            --checkpoints_total_limit=2
                            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8"},
            )
