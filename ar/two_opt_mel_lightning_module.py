"""
ARMel Two-Optimizer Lightning Module (simplified version without AE/VICReg)
 - Manual optimization with 2 optimizers
 - One LLM step + (k-1) diffusion steps per batch
"""

import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

from ar.qwen_tokenizer import get_qwen_tokenizer
from ar.armel import ARMel
from ar.armel_config import ARMelConfig
from ar.qwen import Qwen3LM
from rfwave.mel_processor import MelProcessor, MelConfig
from rfwave.resample import ResampleModule
from rfwave.estimator import RFBackbone
from rfwave.mel_model import RFMelConfig
from utils.logger import get_logger
from utils.config_utils import get_rfmel_config_defaults
from utils.save_code import save_code

try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__)


class ManualOptimMixin:
    """Helper mixin for manual optimization utilities."""

    def _clip_optimizer_params(self, optimizer):
        max_norm = float(self.cfg.training.max_grad_norm)
        if max_norm is None or max_norm <= 0:
            return
        opt = getattr(optimizer, 'optimizer', optimizer)
        params = []
        for group in getattr(opt, 'param_groups', []):
            for p in group.get('params', []):
                if p.grad is not None:
                    params.append(p)
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm)

    def _maybe_unscale(self, optimizer):
        opt = getattr(optimizer, 'optimizer', optimizer)
        strategy = getattr(self.trainer, 'strategy', None)
        precision_plugin = getattr(strategy, 'precision_plugin', None)
        scaler = getattr(precision_plugin, 'scaler', None)
        if scaler is not None:
            scaler.unscale_(opt)

    def _amp_step(self, optimizer):
        strategy = getattr(self.trainer, 'strategy', None)
        precision_plugin = getattr(strategy, 'precision_plugin', None)
        scaler = getattr(precision_plugin, 'scaler', None)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    def _step_raw(self, optimizer, clip: bool = True):
        self._maybe_unscale(optimizer)
        if clip:
            self._clip_optimizer_params(optimizer)
        self._amp_step(optimizer)


class ARMelTwoOptLightningModule(ManualOptimMixin, LightningModule):
    """Two-optimizer training for Mel pipeline: 1 LLM step + diffusion_extra_steps per batch."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.diffusion_extra_steps = max(0, int(cfg.training.get('diffusion_extra_steps', 1)))
        self.save_hyperparameters()

        self.tokenizer = get_qwen_tokenizer(cfg.model.llm_model_path)
        self.head_config = self._create_mel_head_config()
        self.armel_config = self._create_armel_config()
        self.model = self._create_model()
        self.monitor_resample = bool(self.cfg.training.get('monitor_resample', False))
        self.monitor_every_n_steps = int(self.cfg.training.get('monitor_every_n_steps', 100))
        self.automatic_optimization = False

    def _create_mel_head_config(self) -> MelConfig:
        return MelConfig(
            sample_rate=self.cfg.model.mel.sample_rate,
            n_fft=self.cfg.model.mel.n_fft,
            hop_length=self.cfg.model.mel.hop_length,
            n_mels=self.cfg.model.mel.n_mels,
            padding=self.cfg.model.mel.get('padding', 'same'),
        )

    def _create_armel_config(self) -> ARMelConfig:
        return ARMelConfig(
            audio_out_token=self.cfg.arwave.audio_out_token,
            audio_out_bos_token=self.cfg.arwave.audio_out_bos_token,
            audio_eos_token=self.cfg.arwave.audio_eos_token,
            audio_out_token_id=self.tokenizer.convert_tokens_to_ids(
                self.cfg.arwave.audio_out_token
            ),
            audio_out_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                self.cfg.arwave.audio_out_bos_token
            ),
            audio_eos_token_id=self.tokenizer.convert_tokens_to_ids(
                self.cfg.arwave.audio_eos_token
            ),
            ignore_index=self.cfg.arwave.ignore_index,
            round_to=self.cfg.arwave.round_to,
            max_tokens=self.cfg.dataset.max_tokens,
            sample_rate=self.cfg.dataset.sample_rate,
            patch_size=self.cfg.model.patch_size,
            use_skip_connection=bool(self.cfg.model.get('use_skip_connection', False))
        )

    def _create_model(self) -> ARMel:
        attn_implementation = self.cfg.model.get('attn_implementation', 'sdpa')
        llm = Qwen3LM(
            pretrain_path=self.cfg.model.llm_model_path,
            attn_implementation=attn_implementation,
            use_linear_cross_entropy=False
        )
        llm.resize_token_embeddings(len(self.tokenizer))

        mel_processor = MelProcessor(self.head_config)
        comp_spec_dim = self.head_config.n_mels
        llm_hidden_dim = llm.model.config.hidden_size

        rfmel_cfg = self.cfg.model.get('rfmel', {})
        rfmel_defaults = get_rfmel_config_defaults()
        num_prefix_patches = int(rfmel_cfg.get('num_prefix_patches', rfmel_defaults['num_prefix_patches']))

        est_hidden_dim = self.cfg.model.estimator.hidden_dim
        est_intermediate_dim = self.cfg.model.estimator.intermediate_dim
        est_num_layers = self.cfg.model.estimator.num_layers
        estimator = RFBackbone(
            input_channels=comp_spec_dim,
            output_channels=comp_spec_dim,
            dim=est_hidden_dim,
            intermediate_dim=est_intermediate_dim,
            num_layers=est_num_layers,
            prev_cond=bool(num_prefix_patches > 0),
        )

        rs_cfg = self.cfg.model.resample
        resample_module = ResampleModule(
            complex_spec_dim=comp_spec_dim,
            llm_hidden_dim=llm_hidden_dim,
            patch_size=self.cfg.model.patch_size,
            resample_type=rs_cfg.get('resample_type', 'conv'),
            hidden_dims=rs_cfg.get('hidden_dims', 512),
            downsample_strides=rs_cfg.get('downsample_strides', [2, 4]),
            perceiver_depth=rs_cfg.get('perceiver_depth', 6),
            perceiver_heads=rs_cfg.get('perceiver_heads', 8),
            dim_scale_per_stage=rs_cfg.get('dim_scale_per_stage', 1.0),
        )

        rfmel_config = RFMelConfig(
            solver=rfmel_cfg.get('solver', rfmel_defaults['solver']),
            noise_std=rfmel_cfg.get('noise_std', rfmel_defaults['noise_std']),
            t_scheduler=rfmel_cfg.get('t_scheduler', rfmel_defaults['t_scheduler']),
            training_cfg_rate=rfmel_cfg.get('training_cfg_rate', rfmel_defaults['training_cfg_rate']),
            inference_cfg_rate=rfmel_cfg.get('inference_cfg_rate', rfmel_defaults['inference_cfg_rate']),
            n_timesteps=rfmel_cfg.get('n_timesteps', rfmel_defaults['n_timesteps']),
            batch_mul=rfmel_cfg.get('batch_mul', rfmel_defaults['batch_mul']),
            patch_size=self.cfg.model.patch_size,
            num_prefix_patches=num_prefix_patches,
        )

        armel = ARMel(
            config=self.armel_config,
            llm=llm,
            mel_processor=mel_processor,
            estimator=estimator,
            resample_module=resample_module,
            rfmel_config=rfmel_config,
        )
        armel.mel_processor.to(torch.float32)
        return armel

    def compute_loss(self, batch):
        return self.model.compute_loss(
            input_ids=batch.input_ids,
            label_ids=batch.label_ids,
            waveform=batch.waveform,
            waveform_patch_start=batch.waveform_patch_start,
            attention_mask=batch.attention_mask
        )

    def _run_extra_diffusion_steps(self, target, hidden, skip_features, optimizer):
        """Run extra diffusion steps with chunked data."""
        if self.diffusion_extra_steps <= 0:
            return None

        last_loss = None
        patch_size = self.model.config.patch_size
        total_patches = hidden.size(2)
        chunk_size = (total_patches + self.diffusion_extra_steps - 1) // max(1, self.diffusion_extra_steps)

        for i in range(self.diffusion_extra_steps):
            start_patch = i * chunk_size
            end_patch = min(start_patch + chunk_size, total_patches)
            
            hidden_chunk = hidden[..., start_patch:end_patch]
            skip_chunk = skip_features[..., start_patch:end_patch] if skip_features is not None else None
            
            start_frame = start_patch * patch_size
            end_frame = end_patch * patch_size
            target_chunk = target[..., start_frame:end_frame]
            
            loss = self.model.compute_diffusion_loss_from_hidden(
                target_chunk, 
                hidden_chunk, 
                skip_features=skip_chunk
            )
            self.manual_backward(loss)
            self._step_raw(optimizer, clip=True)
            optimizer.zero_grad(set_to_none=True)
            last_loss = loss
            
        return last_loss

    def training_step(self, batch, batch_idx):
        opt1_pl, _ = self.optimizers()
        _, opt2_raw = self.optimizers(use_pl_optimizer=False)

        hidden = self.model.compute_hidden(
            input_ids=batch.input_ids,
            waveform=batch.waveform,
            waveform_patch_start=batch.waveform_patch_start,
            attention_mask=batch.attention_mask
        )

        waveform_hidden_state_detached = hidden['waveform_hidden_state'].detach()
        target = hidden['target_comp_spec']
        skip_features = hidden.get('waveform_emb', None) if self.model.use_skip_connection else None
        skip_features_detached = skip_features.detach() if skip_features is not None else None
        
        mel_loss_extra = self._run_extra_diffusion_steps(
            target, 
            waveform_hidden_state_detached, 
            skip_features_detached, 
            opt2_raw
        )

        # Joint step: LLM loss + diffusion loss
        llm_loss = self.model.compute_llm_loss_from_hidden(hidden['last_hidden_state'], batch.label_ids)
        mel_loss = self.model.compute_diffusion_loss_from_hidden(
            hidden['target_comp_spec'], hidden['waveform_hidden_state'], skip_features=skip_features)

        train_loss = llm_loss + mel_loss
        self.manual_backward(train_loss)
        opt1_pl.step()
        opt1_pl.zero_grad(set_to_none=True)
        self._step_raw(opt2_raw, clip=True)
        opt2_raw.zero_grad(set_to_none=True)

        # Logging
        self.log('train/loss', train_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/llm_loss', llm_loss, on_step=True, on_epoch=False)
        self.log('train/mel_loss', mel_loss, on_step=True, on_epoch=False)
        if mel_loss_extra is not None:
            self.log('train/mel_loss_extra', mel_loss_extra, on_step=True, on_epoch=False)
        self._maybe_log_resample_metrics(stage='train', batch=batch, hidden=hidden)

        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

    def on_before_optimizer_step(self, optimizer):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.cfg.training.max_grad_norm,
            gradient_clip_algorithm="norm",
        )

    def on_fit_start(self):
        if self.global_rank == 0:
            save_dir = self.cfg.training.get("log_dir", None)
            if save_dir:
                logger.info(f"Saving code to {save_dir}")
                try:
                    save_code(None, save_dir)
                    logger.info("Code saved successfully")
                except Exception as e:
                    logger.error(f"Failed to save code: {e}")

    def validation_step(self, batch, batch_idx):
        loss_dict = self.compute_loss(batch)
        bs = int(batch.input_ids.size(0))
        self.log('val/loss', loss_dict['loss'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log('val/llm_loss', loss_dict['llm_loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log('val/mel_loss', loss_dict['mel_loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        self._maybe_log_resample_metrics(stage='val', batch=batch, batch_idx=batch_idx)
        return loss_dict['loss']

    def configure_optimizers(self):
        base_lr = self.cfg.training.learning_rate
        diffusion_lr_mul = 1.

        def params_set(mod):
            if mod is None:
                return {}
            return {id(p): p for p in mod.parameters()}

        llm = self.model.llm
        rs = self.model.resample_module
        rf = self.model.rfmel
        est = self.model.estimator

        opt1_ids = {}
        for m in (llm, rs):
            opt1_ids.update(params_set(m))

        opt2_ids = {}
        for m in (rf, est):
            opt2_ids.update(params_set(m))

        for pid in list(opt1_ids.keys()):
            opt2_ids.pop(pid, None)

        def is_no_decay(name: str) -> bool:
            s = name.lower()
            return ('bias' in name) or ('norm' in s) or ('ln' in s)

        opt1_decay_params = []
        opt1_no_decay_params = []
        opt2_params = []
        unassigned_params = []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in opt1_ids:
                if is_no_decay(n):
                    opt1_no_decay_params.append(p)
                else:
                    opt1_decay_params.append(p)
            elif pid in opt2_ids:
                opt2_params.append(p)
            else:
                unassigned_params.append(n)

        if unassigned_params:
            raise ValueError(
                f"Fool-proof check failed: {len(unassigned_params)} trainable parameters "
                f"are NOT assigned to any optimizer:\n{unassigned_params}"
            )

        weight_decay = self.cfg.training.weight_decay
        opt1_groups = []
        if opt1_decay_params:
            opt1_groups.append({'params': opt1_decay_params, 'weight_decay': weight_decay})
        if opt1_no_decay_params:
            opt1_groups.append({'params': opt1_no_decay_params, 'weight_decay': 0.0})
        opt1 = torch.optim.AdamW(opt1_groups, lr=base_lr)

        opt2_groups = []
        if opt2_params:
            opt2_groups.append({'params': opt2_params, 'weight_decay': 0.0})
        opt2 = torch.optim.AdamW(opt2_groups, lr=base_lr * diffusion_lr_mul)

        max_steps = self.cfg.training.max_steps
        warmup_steps = self.cfg.training.get('warmup_steps', 0)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        sch1 = torch.optim.lr_scheduler.LambdaLR(opt1, lr_lambda)
        sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda)

        def count_tensors(params):
            return len(list(params))
        def count_numel(params):
            return sum(p.numel() for p in params)

        all_trainable = [p for p in self.model.parameters() if p.requires_grad]
        total_tensors = count_tensors(all_trainable)
        total_params = count_numel(all_trainable)
        opt1_all = opt1_decay_params + opt1_no_decay_params
        opt1_tensors = count_tensors(opt1_all)
        opt1_params_numel = count_numel(opt1_all)
        opt2_tensors = count_tensors(opt2_params)
        opt2_params_numel = count_numel(opt2_params)

        logger.info("=" * 80)
        logger.info("Two-Opt Optimizer Parameter Summary (Mel):")
        logger.info(f"  - Total trainable tensors: {total_tensors}, params: {total_params:,}")
        logger.info(f"  - Opt1 tensors: {opt1_tensors}, params: {opt1_params_numel:,} (LLM + Resample)")
        logger.info(f"  - Opt2 tensors: {opt2_tensors}, params: {opt2_params_numel:,} (RFMel + estimator)")
        logger.info("=" * 80)

        return [
            {'optimizer': opt1, 'lr_scheduler': {'scheduler': sch1, 'interval': 'step', 'frequency': 1}},
            {'optimizer': opt2, 'lr_scheduler': {'scheduler': sch2, 'interval': 'step', 'frequency': 1}}
        ]

    def _maybe_log_resample_metrics(self, stage: str, batch, hidden=None, batch_idx: int = None):
        if not self.monitor_resample:
            return
        if stage == 'train':
            if self.global_step % max(1, self.monitor_every_n_steps) != 0:
                return
        elif stage == 'val':
            if batch_idx != 0:
                return
        else:
            return

        if hidden is None:
            hidden = self.model.compute_hidden(
                input_ids=batch.input_ids,
                waveform=batch.waveform,
                waveform_patch_start=batch.waveform_patch_start,
                attention_mask=batch.attention_mask
            )
        norms = self.model.get_resample_io_norms(
            ds_in=hidden['target_comp_spec'],
            up_in=hidden['waveform_hidden_state'],
        )
        for k, v in norms.items():
            self.log(f'{stage}/{k}', v, on_step=True, on_epoch=False)

    def on_validation_epoch_end(self):
        """Run inference test and log audio to wandb."""
        if not self.cfg.training.get('run_inference_test', False):
            return
        if self.trainer is None or not self.trainer.is_global_zero:
            return

        experiment = getattr(self.logger, "experiment", None) if self.logger else None
        has_wandb = wandb is not None and experiment is not None and hasattr(experiment, "log")
        if not has_wandb:
            return

        try:
            from ar.inference_test import run_armel_inference_test

            logger.info("Running inference test...")
            with self.trainer.precision_plugin.forward_context():
                waveform, reconstructed_prompt = run_armel_inference_test(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    ref_audio=self.cfg.training.get('inference_ref_audio', 'fanren08'),
                    transcript=self.cfg.training.get('inference_transcript', 'fanren_short.txt'),
                    max_new_tokens=self.cfg.training.get('inference_max_new_tokens', 256),
                )

            sr = self.model.config.sample_rate
            if waveform is not None:
                waveform_np = waveform.float().squeeze(0).cpu().numpy()
                experiment.log({
                    "val/inference_audio": wandb.Audio(
                        waveform_np, sample_rate=sr, caption=f"step_{self.global_step}"
                    )
                }, commit=False)
                logger.info(f"Logged inference audio (duration: {len(waveform_np) / sr:.2f}s)")

            if reconstructed_prompt is not None:
                prompt_np = reconstructed_prompt.float().squeeze(0).cpu().numpy()
                experiment.log({
                    "val/reconstructed_prompt": wandb.Audio(
                        prompt_np, sample_rate=sr, caption=f"step_{self.global_step}"
                    )
                }, commit=False)
        except Exception as e:
            logger.warning(f"Inference test failed: {e}")
