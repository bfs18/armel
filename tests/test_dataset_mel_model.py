import torch

from dataset.arwave_dataset import ARWaveDataset
from dataset.data_collate import SampleCollator
from ar.qwen_tokenizer import get_qwen_tokenizer
from ar.special_tokens import audio_out_bos_token, audio_out_token, audio_eos_token, pad_token
from ar.armel import ARMel
from ar.armel_config import ARMelConfig
from ar.qwen import Qwen3LM
from ar.mel_generate import generate, decode_one_patch
from ar.generate import get_prompt_embedding
from rfwave.mel_processor import MelProcessor, MelConfig
from rfwave.estimator import RFBackbone
from rfwave.mel_model import RFMelConfig
from rfwave.resample import ResampleModule
from utils.data import place_data


def test_dataset():
    dataset_path = "/data/corpus/arwave-trainset"
    qwen_model_path = "Qwen3-0.6B"
    tokenizer = get_qwen_tokenizer(qwen_model_path)

    # Sanity check: tokens present
    _ = tokenizer.encode(
        f"{audio_out_bos_token}{audio_out_token}{audio_eos_token}",
        add_special_tokens=False,
    )

    audio_out_bos_token_id, audio_out_token_id, audio_eos_token_id, pad_token_id = tokenizer.convert_tokens_to_ids(
        [audio_out_bos_token, audio_out_token, audio_eos_token, pad_token]
    )

    mel_config = MelConfig(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=120)
    patch_size = 8

    dataset = ARWaveDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        sample_rate=24000,
        patch_size=patch_size,
        head_config=mel_config,
        max_tokens=1024,
    )
    collator = SampleCollator(
        audio_out_token_id=audio_out_token_id,
        pad_token_id=pad_token_id,
        head_config=mel_config,
        patch_size=patch_size,
    )

    samples = []
    dataset_iter = iter(dataset)
    for _ in range(4):
        sample_i = next(dataset_iter)
        samples.append(sample_i)

    batch_data = collator(samples)
    print(batch_data.input_ids.shape)
    print(batch_data.waveform.shape if batch_data.waveform is not None else None)
    print(batch_data.waveform_patch_start)
    return batch_data


def test_model(batch_data):
    device = "cuda"
    dtype = torch.bfloat16

    qwen_model_path = "Qwen3-0.6B"
    tokenizer = get_qwen_tokenizer(qwen_model_path)
    audio_out_bos_token_id, audio_out_token_id, audio_eos_token_id, pad_token_id = tokenizer.convert_tokens_to_ids(
        [audio_out_bos_token, audio_out_token, audio_eos_token, pad_token]
    )

    mel_config = MelConfig(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=120)
    rfmel_config = RFMelConfig()
    armel_config = ARMelConfig(
        audio_out_token=audio_out_token,
        audio_out_bos_token=audio_out_bos_token,
        audio_eos_token=audio_eos_token,
        audio_out_token_id=audio_out_token_id,
        audio_out_bos_token_id=audio_out_bos_token_id,
        audio_eos_token_id=audio_eos_token_id,
    )

    llm = Qwen3LM(pretrain_path=qwen_model_path, attn_implementation="sdpa").to(device=device)
    llm.resize_token_embeddings(len(tokenizer))
    mel_processor = MelProcessor(mel_config)

    comp_spec_dim = mel_config.n_mels
    llm_hidden_dim = llm.model.config.hidden_size
    est_hidden_dim = 1024
    rs_hidden_dim = 1024
    rs_downsample_strides = [2, 4]
    patch_size = 8

    # IMPORTANT: Data flow for RFBackbone input dimension:
    # 1. ResampleModule.upsample projects LLM hidden states (llm_hidden_dim) to comp_spec_dim
    # 2. This upsampled output serves as conditional input for RFBackbone
    # 3. Therefore, RFBackbone's input_channels must equal comp_spec_dim
    estimator = RFBackbone(
        input_channels=comp_spec_dim,
        output_channels=comp_spec_dim,
        dim=est_hidden_dim,
        intermediate_dim=est_hidden_dim * 4,
        num_layers=4,
    )

    resample_module = ResampleModule(
        complex_spec_dim=comp_spec_dim,
        llm_hidden_dim=llm_hidden_dim,
        patch_size=patch_size,
        hidden_dims=rs_hidden_dim,
        downsample_strides=rs_downsample_strides,
    )

    armel = ARMel(
        config=armel_config,
        llm=llm,
        mel_processor=mel_processor.to(device=device, dtype=dtype),
        estimator=estimator.to(device=device, dtype=dtype),
        resample_module=resample_module.to(device=device, dtype=dtype),
        rfmel_config=rfmel_config,
    )
    armel.rfmel.to(device=device, dtype=dtype)
    armel.mel_processor.to(torch.float32)

    out = armel.compute_loss(
        input_ids=batch_data.input_ids,
        label_ids=batch_data.label_ids,
        waveform=batch_data.waveform,
        waveform_patch_start=batch_data.waveform_patch_start,
        attention_mask=batch_data.attention_mask,
    )

    prompt_emb = get_prompt_embedding(
        armel, batch_data.input_ids, batch_data.waveform, batch_data.waveform_patch_start)
    res = generate(
        model=armel,
        prompt_emb=prompt_emb[:1],
        decode_func=decode_one_patch,
        max_new_tokens=512,
        audio_eos_id=audio_eos_token_id,
        top_p=0.7,
        temperature=0.7
    )
    print(res[0].shape)

    return out


if __name__ == "__main__":
    batch = test_dataset()
    batch = place_data(batch, device="cuda", dtype=torch.bfloat16)
    result = test_model(batch)
    print(result)
