import math
import torch
import numpy as np
import pydub
import random

from torch.utils.data import Dataset
from datasets import load_from_disk
from dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from dataset.data_types import Message, AudioContent, ChatMLSample
from utils.audio import read_audio_from_base64, concat_waveforms_with_offsets


class ARWaveDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, sample_rate, head_config, patch_size, max_tokens=2048):
        """
        Args:
            dataset_path (str): Path to the directory where the processed dataset was saved.
            tokenizer: The text tokenizer.
            max_tokens (int): The maximum number of audio tokens for a single sample.
        """
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.dataset = None  # Defer loading to __getitem__
        self.sample_rate = sample_rate
        self.head_config = head_config
        self.patch_size = patch_size

    def _load_dataset(self):
        """Loads the dataset if it hasn't been loaded in the current process."""
        if self.dataset is None:
            self.dataset = load_from_disk(self.dataset_path)

    def __len__(self):
        """
        Returns the total number of podcasts in the dataset.
        """
        self._load_dataset()
        return len(self.dataset)

    def _get_audio_num_patches(self, duration, sr):
        waveform_length = int(duration * sr)
        total_frames = waveform_length // self.head_config.hop_length + (1 if self.head_config.padding == "center" else 0)
        num_patches = math.ceil(total_frames / self.patch_size)
        return num_patches

    def __getitem__(self, idx):
        """
        Retrieves a sample from a podcast, starting from a random clip and including
        as many subsequent clips as possible up to `max_tokens`. The sample is
        formatted into a `ChatMLDatasetSample`.

        Args:
            idx (int): The index of the podcast to retrieve.

        Returns:
            ChatMLDatasetSample: A sample ready to be processed by the collator.
        """
        self._load_dataset()
        podcast = self.dataset[idx]
        clips = podcast['processed_clips']
        assert len(clips) > 0, "Podcast has no clips"

        # 1. Tokenize all clips once and store lengths and token IDs.

        # Based on `prepare_chatml_sample`, the overhead for a user text message + assistant audio message pair is:
        # User: <|start_header_id|>user<|end_header_id|>\n\n (4) + text + <|eot_id|> (1) = 5 tokens
        # Assistant: <|start_header_id|>assistant<|end_header_id|>\n\n (4) + <|audio_out_bos|><|AUDIO_OUT|><|audio_eos|> (3) + <|eot_id|> (1) = 8 tokens
        # Total overhead per pair = 5 + 8 = 13 tokens.
        CHATML_PAIR_OVERHEAD = 13

        clip_info = []
        for clip in clips:
            if "raw_audio" in clip and clip["raw_audio"] is not None:
                duration = clip['audio_duration']
                num_patches = self._get_audio_num_patches(duration, self.sample_rate)
                text_tokens = self.tokenizer(clip["text"], add_special_tokens=False)['input_ids']
                num_text_tokens = len(text_tokens)
                clip_info.append({
                    "total_length": num_patches + num_text_tokens + CHATML_PAIR_OVERHEAD,
                    "text_tokens": text_tokens,
                    "num_text_tokens": num_text_tokens
                })
            else:
                clip_info.append({
                    "total_length": 0,
                    "text_tokens": [],
                    "num_text_tokens": 0
                })

        clip_token_lengths = [info['total_length'] for info in clip_info]

        # 2. Calculate the total tokens remaining from each possible start point.
        cumulative_lengths_from_end = np.cumsum(clip_token_lengths[::-1])[::-1]

        # 3. Identify all clips that are valid starting points.
        # A clip is valid if the sequence from it to the end is long enough.
        possible_start_indices = np.where(cumulative_lengths_from_end >= self.max_tokens)[0]

        # 4. Choose a starting clip.
        if len(possible_start_indices) > 0:
            # If there are valid start points, choose one randomly.
            start_clip_idx = random.choice(possible_start_indices)
        else:
            # If the entire podcast is shorter than max_tokens, just start from the beginning.
            start_clip_idx = 0

        selected_clips_data = []
        total_length = 0

        for i in range(start_clip_idx, len(clips)):
            clip = clips[i]
            info = clip_info[i]

            if "raw_audio" in clip and clip["raw_audio"] is not None:
                # Reuse the pre-calculated text token info
                clip_total_length = info['total_length']

                if (total_length + clip_total_length) > self.max_tokens and selected_clips_data:
                    break

                waveform, sr = read_audio_from_base64(
                    clip['raw_audio'], target_sr=self.sample_rate, audio_path=clip['audio_path'])
                selected_clips_data.append({
                    "text": clip["text"],
                    "waveform": waveform,
                    "sr": sr
                })
                total_length += clip_total_length
        # Create ChatML messages
        messages = []
        waveform_l = []
        for clip_data in selected_clips_data:
            messages.append(Message(role='user', content=clip_data['text']))
            # AudioContent is a placeholder; the actual audio comes from pre-computed tokens
            messages.append(Message(role='assistant', content=AudioContent(audio_url="placeholder.wav")))
            waveform_l.append(clip_data['waveform'])

        # Randomly select the number of pairs to use as context for voice cloning
        if len(selected_clips_data) > 0:
            num_context_pairs = random.randint(0, len(selected_clips_data) - 1)
        else:
            num_context_pairs = 0

        # start_index = 2 * num_context_pairs
        start_index = 0

        # Concatenate audio tokens
        waveform_concat, waveform_start = concat_waveforms_with_offsets(waveform_l)

        chatml_sample = ChatMLSample(messages=messages, start_index=start_index)
        input_tokens, label_tokens, *_ = prepare_chatml_sample(chatml_sample, self.tokenizer)
        audio_sample_rate = torch.tensor([self.sample_rate] * len(waveform_l), dtype=torch.long)

        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=torch.LongTensor(label_tokens),
            audio_waveforms_concat=waveform_concat,
            audio_waveforms_start=waveform_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=None,
        )
        return sample
