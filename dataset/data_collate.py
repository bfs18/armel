import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Optional, Union

from dataset.chatml_dataset import ChatMLDatasetSample
from rfwave.mel_processor import MelConfig


def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to


def _ceil_to_next_power_of_two(self, x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


@dataclass
class BatchInput:
    input_ids: torch.LongTensor  # shape (bsz, seq_len).
    attention_mask: torch.Tensor  # shape (bsz, seq_len).
    waveform: Optional[torch.Tensor]  # shape (1, waveform_length)
    waveform_patch_start: Optional[torch.LongTensor]  # shape (num_audio_out,)
    waveform_patch_start_group_loc: Optional[torch.LongTensor]  # shape (num_audio_out,)
    label_ids: Optional[torch.LongTensor]  # shape (bsz, seq_len)


class SampleCollator(object):
    """Sample collator for ARMel training."""

    def __init__(
        self,
        audio_out_token_id,
        pad_token_id,
        head_config: MelConfig,
        patch_size: int,
        round_to=8,
        ignore_index=-100
    ):
        assert head_config.padding in ["same", "causal"], "Only same or causal padding is supported"
        self.round_to = round_to
        self.audio_out_token_id = audio_out_token_id
        self.pad_token_id = pad_token_id
        self.head_config = head_config
        self.patch_size = patch_size
        self.ignore_index = ignore_index

    def _prepare_waveform_for_patches(self, waveform):
        waveform = waveform[..., :waveform.shape[-1] // self.head_config.hop_length * self.head_config.hop_length]
        waveform_length = waveform.shape[-1]
        total_frames = waveform_length // self.head_config.hop_length + (1 if self.head_config.padding == "center" else 0)
        num_patches = math.ceil(total_frames / self.patch_size)
        padded_frames = num_patches * self.patch_size
        padded_waveform_length = (padded_frames - (1 if self.head_config.padding == 'center' else 0)) * self.head_config.hop_length
        padding_length = padded_waveform_length - waveform_length
        if padding_length > 0:
            waveform = torch.nn.functional.pad(waveform, (padding_length // 2, padding_length - padding_length // 2))
        return waveform, num_patches

    def _process_and_duplicate_audio_out_tokens(
            self, input_ids: torch.Tensor, label_ids: Optional[torch.Tensor],
            audio_out_token_id: int, waveform_num_patches: torch.Tensor) -> (torch.Tensor, Optional[torch.Tensor]):
        audio_out_token_mask = input_ids == audio_out_token_id
        token_placeholder_num = torch.ones_like(input_ids)
        token_placeholder_num[audio_out_token_mask] = waveform_num_patches.long()
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        new_token_num = token_placeholder_num.sum().int()
        new_input_ids = input_ids.new_zeros((new_token_num, ))
        seq_indices = torch.arange(new_token_num, device=input_ids.device)
        waveform_patch_ends = new_token_positions[audio_out_token_mask]
        waveform_patch_starts = waveform_patch_ends - waveform_num_patches + 1
        waveform_patch_mask = ((seq_indices.unsqueeze(1) >= waveform_patch_starts.unsqueeze(0)) &
                               (seq_indices.unsqueeze(1) <= waveform_patch_ends.unsqueeze(0))).any(dim=1)
        
        new_input_ids[waveform_patch_mask] = audio_out_token_id
        new_input_ids[~waveform_patch_mask] = input_ids[~audio_out_token_mask]
        
        new_label_ids = None
        if label_ids is not None:
            new_label_ids = input_ids.new_full((new_token_num, ), self.ignore_index)
            new_label_ids[waveform_patch_mask] = audio_out_token_id
            new_label_ids[~waveform_patch_mask] = label_ids[~audio_out_token_mask]
        
        return new_input_ids, new_label_ids

    def __call__(self, batch: List[ChatMLDatasetSample]):
        """Collate the input data with support for long audio processing."""

        label_ids = None
        if all([ele.label_ids is None for ele in batch]):
            return_labels = False
        else:
            return_labels = True

        input_ids_l = []
        label_ids_l = []
        waveform_l = []
        waveform_num_patches_l = []
        waveform_patch_start_group_loc_l = []

        for i in range(len(batch)):
            sample = batch[i]
            audio_out_mask = sample.input_ids == self.audio_out_token_id
            audio_out_indices = torch.where(audio_out_mask)[0]
            waveform_l_i = []
            waveform_num_patches_l_i = []

            if sample.audio_waveforms_concat is not None:
                assert len(audio_out_indices) == len(sample.audio_waveforms_start), \
                    f"Mismatch between audio_out tokens and waveforms in batch[{i}]"
                for idx, audio_idx in enumerate(audio_out_indices):
                    wv, sr = sample.get_waveform(idx)
                    wv, wv_num_patches = self._prepare_waveform_for_patches(wv)
                    waveform_l_i.append(wv)
                    waveform_num_patches_l_i.append(wv_num_patches)
                    waveform_patch_start_group_loc_l.append(i)

            waveform_num_patches = torch.tensor(waveform_num_patches_l_i)
            new_input_ids, new_label_ids = self._process_and_duplicate_audio_out_tokens(
                sample.input_ids, sample.label_ids, self.audio_out_token_id, waveform_num_patches)
            input_ids_l.append(new_input_ids)
            label_ids_l.append(new_label_ids)
            waveform_l.extend(waveform_l_i)
            waveform_num_patches_l.extend(waveform_num_patches_l_i)

        max_seq_length = max([sample.shape[-1] for sample in input_ids_l])
        max_seq_length = _ceil_to_nearest(max_seq_length, self.round_to)

        input_ids = torch.stack(
            [F.pad(input_ids_i, (0, max_seq_length - input_ids_i.shape[-1]), value=self.pad_token_id)
             for input_ids_i in input_ids_l], dim=0)
        attention_mask = torch.stack(
            [F.pad(torch.ones_like(input_ids_i), (0, max_seq_length - input_ids_i.shape[-1]), value=0)
             for input_ids_i in input_ids_l], dim=0)
        
        if return_labels:
            label_ids = torch.stack(
                [F.pad(label_ids_i, (0, max_seq_length - label_ids_i.shape[-1]), value=self.ignore_index)
                 for label_ids_i in label_ids_l], dim=0)
        else:
            label_ids = None
        
        if len(waveform_l) > 0:
            waveform_patch_start = torch.cumsum(torch.tensor([0] + waveform_num_patches_l[:-1]), dim=-1)
            waveform_patch_start_group_loc = torch.tensor(waveform_patch_start_group_loc_l)
            waveform = torch.cat(waveform_l, dim=-1)
        else:
            waveform_patch_start = None
            waveform_patch_start_group_loc = None
            waveform = None

        return BatchInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            waveform=waveform,
            waveform_patch_start=waveform_patch_start,
            waveform_patch_start_group_loc=waveform_patch_start_group_loc,
            label_ids=label_ids
        )
