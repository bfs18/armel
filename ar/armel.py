"""
ARMel - AR + Mel hybrid model (simplified version without AE/VICReg).

This is the isolated version for ar-dit-mel project.
"""
from typing import Dict, Tuple
import torch
from torch import nn
from ar.armel_config import ARMelConfig
from rfwave.mel_model import RFMel, RFMelConfig
from rfwave.mel_processor import MelProcessor
from rfwave.resample import ResampleModule
from ar.qwen import Qwen3LM


class ARMel(nn.Module):
    def __init__(
        self,
        config: ARMelConfig,
        llm: Qwen3LM,
        mel_processor: MelProcessor,
        estimator: nn.Module,
        resample_module: ResampleModule,
        rfmel_config: RFMelConfig,
    ):
        super().__init__()
        self.config = config
        self.llm = llm
        self.mel_processor = mel_processor
        self.resample_module = resample_module
        self.estimator = estimator
        self.use_skip_connection = config.use_skip_connection
        
        # Initialize RFMel
        self.rfmel = RFMel(
            estimator=estimator,
            mel_processor=mel_processor,
            resample_module=resample_module,
            config=rfmel_config,
            use_skip_connection=config.use_skip_connection,
        )

        # Set audio_out_token_id from tokenizer if not provided
        if self.config.audio_out_token_id is None and hasattr(llm, 'tokenizer'):
            self.config.audio_out_token_id = llm.tokenizer.convert_tokens_to_ids(
                self.config.audio_out_token
            )
        if self.config.audio_out_bos_token_id is None and hasattr(llm, 'tokenizer'):
            self.config.audio_out_bos_token_id = llm.tokenizer.convert_tokens_to_ids(
                self.config.audio_out_bos_token
            )
        if self.config.audio_eos_token_id is None and hasattr(llm, 'tokenizer'):
            self.config.audio_eos_token_id = llm.tokenizer.convert_tokens_to_ids(
                self.config.audio_eos_token
            )

    def get_text_embedding(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Convert text tokens to embeddings using LLM's input embedding layer."""
        return self.llm.model.get_input_embeddings()(text_tokens)

    def get_waveform_embedding(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        comp_spec = self.mel_processor.get_norm_mel(waveform)
        waveform_emb = self.resample_module.downsample(comp_spec)
        return comp_spec, waveform_emb

    def _merge_waveform_to_text(self, text_tokens, waveform_emb, waveform_patch_start):
        """Merge waveform embeddings into text token sequence."""
        waveform_out_mask = (text_tokens == self.config.audio_out_token_id)
        text_emb = self.get_text_embedding(text_tokens)
        waveform_emb = waveform_emb.squeeze(0).T
        text_emb[waveform_out_mask] = waveform_emb.to(text_emb.dtype)
        return text_emb

    def _get_waveform_hidden_state(self, text_tokens, hidden_state):
        """Extract hidden states for autoregressive waveform prediction."""
        bos_mask = text_tokens == self.config.audio_out_bos_token_id
        audio_mask = text_tokens == self.config.audio_out_token_id
        eos_mask = text_tokens == self.config.audio_eos_token_id
        
        waveform_out_mask = torch.logical_or(bos_mask, audio_mask)
        eos_mask_shifted = torch.nn.functional.pad(eos_mask, [0, 1])[..., 1:]
        waveform_out_mask = torch.logical_and(waveform_out_mask, torch.logical_not(eos_mask_shifted))
        
        hidden_state = hidden_state[waveform_out_mask]
        return hidden_state.T.unsqueeze(0)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        label_ids: torch.Tensor,
        waveform: torch.Tensor,
        waveform_patch_start: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """Forward pass for the ARMel model."""
        hidden = self.compute_hidden(
            input_ids=input_ids,
            waveform=waveform,
            waveform_patch_start=waveform_patch_start,
            attention_mask=attention_mask
        )
        
        llm_loss = self.compute_llm_loss_from_hidden(hidden['last_hidden_state'], label_ids)
        
        skip_features = hidden.get('waveform_emb', None) if self.use_skip_connection else None
        mel_loss = self.compute_diffusion_loss_from_hidden(
            hidden['target_comp_spec'], 
            hidden['waveform_hidden_state'],
            skip_features=skip_features
        )

        loss = llm_loss + mel_loss

        return {
            'loss': loss,
            'llm_loss': llm_loss,
            'mel_loss': mel_loss,
        }

    def compute_hidden(
        self,
        input_ids: torch.Tensor,
        waveform: torch.Tensor,
        waveform_patch_start: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute hidden states and targets used by both losses."""
        comp_spec, waveform_emb = self.get_waveform_embedding(waveform)
        input_embeddings = self._merge_waveform_to_text(input_ids, waveform_emb, waveform_patch_start)
        llm_output = self.llm.forward_lm(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        last_hidden_state = llm_output.last_hidden_state
        waveform_hidden_state = self._get_waveform_hidden_state(input_ids, last_hidden_state)

        num_predictions = waveform_hidden_state.size(2)
        num_targets = comp_spec.size(2)
        assert num_predictions * self.config.patch_size == num_targets

        return {
            'last_hidden_state': last_hidden_state,
            'waveform_hidden_state': waveform_hidden_state,
            'target_comp_spec': comp_spec,
            'waveform_emb': waveform_emb,
        }

    def compute_llm_loss_from_hidden(
        self,
        last_hidden_state: torch.Tensor,
        label_ids: torch.Tensor,
        ignore_index: int = None,
    ) -> torch.Tensor:
        if ignore_index is None:
            ignore_index = self.config.ignore_index
        return self.llm.forward_lm_head(
            last_hidden_state[:, :-1], label_ids[:, 1:], ignore_index=ignore_index)

    def compute_diffusion_loss_from_hidden(
        self,
        target_comp_spec: torch.Tensor,
        waveform_hidden_state: torch.Tensor,
        skip_features: torch.Tensor = None,
    ) -> torch.Tensor:
        prefix = target_comp_spec[..., :-self.config.patch_size] if self.rfmel.use_prefix else None
        
        if skip_features is not None and self.use_skip_connection:
            skip_features = nn.functional.pad(skip_features, (1, 0))[:, :, :-1]
        
        return self.rfmel.compute_loss(
            target_comp_spec,
            waveform_hidden_state,
            prefix=prefix,
            skip_features=skip_features
        )

    def get_resample_io_norms(self, ds_in: torch.Tensor, up_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute RMS norms for resample I/O paths."""
        with torch.no_grad():
            def rms(x: torch.Tensor) -> torch.Tensor:
                return torch.sqrt(torch.mean(x.float().pow(2)))

            ds_out = self.resample_module.downsample(ds_in.float())
            up_out = self.resample_module.upsample(up_in.float())

            return {
                'resample/ds_in_norm': rms(ds_in),
                'resample/ds_out_norm': rms(ds_out),
                'resample/up_in_norm': rms(up_in),
                'resample/up_out_norm': rms(up_out),
            }

    def get_waveform_embedding_one_step(self, comp_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform_emb = self.resample_module.downsample(comp_spec)
        return comp_spec, waveform_emb

    def forward_generate_ar(
        self,
        emb,
        cache,
        cache_position=None,
        return_all=False
    ):
        outs, new_cache = self.llm.forward_one_step(emb, cache=cache, cache_position=cache_position)
        if emb.size(1) > 1 and not return_all:
            token_logits = outs.logits[:, -1:]
            hidden_states = outs.hidden_states[-1][:, -1:]
        else:
            token_logits = outs.logits
            hidden_states = outs.hidden_states[-1]
        return ({"token_logits": token_logits,
                 "hidden_states": hidden_states},
                new_cache)

    def forward_generate_wave(
        self,
        hidden_states,
        prefix: torch.Tensor = None,
        skip_features: torch.Tensor = None
    ):
        """Generate mel spectrogram from hidden states."""
        comp_spec = self.rfmel.sample(hidden_states, one_patch=True, prefix=prefix, skip_features=skip_features)
        return comp_spec
