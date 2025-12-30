import json
import torch

from typing import Optional
from dataset.data_types import AudioContent, TextContent, Message, ChatMLSample
from dataclasses import dataclass


@dataclass
class ChatMLDatasetSample:
    input_ids: torch.LongTensor  # Shape (seq_len,): The input text tokens.
    label_ids: torch.LongTensor  # Shape (seq_len,): The label ids.
    audio_waveforms_concat: (
        Optional[torch.Tensor]
    )  # Shape (total_wv_length,): The concatenated audio waveforms for audio-in features.
    audio_waveforms_start: (
        Optional[torch.LongTensor]
    )  # Shape (num_audios,): The start index of each audio waveform in the concatenated audio waveforms.
    audio_sample_rate: Optional[torch.Tensor]  # Shape (num_audios,): The sampling rate of the audio waveforms.
    audio_speaker_indices: (
        Optional[torch.LongTensor]
    )  # Shape (num_audios,) -1 means unknown speaker: The speaker indices for each audio.

    def num_audios(self):
        if self.audio_waveforms_start is None:
            return 0
        return len(self.audio_waveforms_start)

    def get_waveform(self, idx):
        if self.audio_waveforms_concat is None:
            raise ValueError("audio_waveforms_concat is None, cannot get waveform")
        wv_start = self.audio_waveforms_start[idx]
        sr = self.audio_sample_rate[idx]
        if idx < len(self.audio_waveforms_start) - 1:
            wv_end = self.audio_waveforms_start[idx + 1]
        else:
            wv_end = self.audio_waveforms_concat.shape[-1]
        return self.audio_waveforms_concat[..., wv_start:wv_end], sr

def prepare_chatml_sample(sample: ChatMLSample, tokenizer):
    """Preprocess the ChatML sample to get the tokens for the text part.

    Args:
        sample (ChatMLSample): The ChatML sample to preprocess.
        tokenizer: The tokenizer to use for encoding the text.

    """

    try:
        input_tokens = []
        label_tokens = []
        audio_contents = []
        speaker_id = None
        if sample.speaker is not None:
            speaker_id = sample.speaker
        elif sample.misc is not None:
            if "speaker" in sample.misc:
                speaker_id = sample.misc["speaker"]

        total_m = len(sample.messages)
        for turn_id, message in enumerate(sample.messages):
            role = message.role
            recipient = message.recipient
            content = message.content
            content_l = []

            if isinstance(content, str):
                content_l.append(TextContent(text=content))
            elif isinstance(content, TextContent):
                content_l.append(content)
            elif isinstance(content, AudioContent):
                content_l.append(content)
            elif isinstance(content, list):
                for ele in content:
                    if isinstance(ele, str):
                        content_l.append(TextContent(text=ele))
                    else:
                        content_l.append(ele)
            if turn_id == 0:
                prefix = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
            else:
                prefix = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            eot_postfix = "<|eot_id|>"
            eom_postfix = "<|eom_id|>"

            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            input_tokens.extend(prefix_tokens)
            label_tokens.extend([-100 for _ in prefix_tokens])

            if recipient:
                assert role == "assistant", "Recipient is only available for assistant role."
                recipient_tokens = tokenizer.encode(f"{recipient}<|recipient|>", add_special_tokens=False)
                input_tokens.extend(recipient_tokens)
                label_tokens.extend(recipient_tokens)

            for content in content_l:
                if content.type == "text":
                    text_tokens = tokenizer.encode(content.text, add_special_tokens=False)
                    input_tokens.extend(text_tokens)
                    if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                        label_tokens.extend(text_tokens)
                    else:
                        label_tokens.extend([-100 for _ in text_tokens])

                elif content.type == "audio":
                    # Generate the text-part of the audio tokens
                    audio_contents.append(content)
                    if role == "user" or role == "system":
                        # Add the text tokens
                        text_tokens = tokenizer.encode(
                            f"<|audio_bos|><|AUDIO|><|audio_eos|>",
                            add_special_tokens=False,
                        )
                        input_tokens.extend(text_tokens)
                        label_tokens.extend([-100 for _ in text_tokens])
                    elif role == "assistant":
                        # Add the text tokens for audio-out part.
                        text_tokens = tokenizer.encode(
                            f"<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>",
                            add_special_tokens=False,
                        )
                        input_tokens.extend(text_tokens)
                        if sample.start_index is None or turn_id >= sample.start_index:
                            label_tokens.extend(text_tokens)
                        else:
                            label_tokens.extend([-100 for _ in text_tokens])
            next_id = turn_id + 1
            if role == "assistant" and next_id != total_m and sample.messages[next_id].role == "assistant":
                postfix_tokens = tokenizer.encode(eom_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            else:
                postfix_tokens = tokenizer.encode(eot_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                label_tokens.extend(postfix_tokens)
            else:
                label_tokens.extend([-100 for _ in postfix_tokens])

        return input_tokens, label_tokens, audio_contents, speaker_id

    except Exception as e:
        print(f"Error in prepare_chatml_sample: {str(e)}")
        print(f"Sample data: {json.dumps(sample, indent=2)}")
        return None, None, None, None
