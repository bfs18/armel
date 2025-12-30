import argparse
import json
import logging
import re

from pathlib import Path
from datasets import Dataset
from tinytag import TinyTag

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.audio import encode_audio_to_base64


def natural_sort_key(s):
    """A key for natural sorting based on the final number in the filename."""
    s = str(s)
    match = re.search(r'_(\d+)\.m4a$', s)
    if match:
        return int(match.group(1))
    return s


def find_podcasts(data_dir):
    """Finds all podcast subdirectories in the main data directory."""
    data_path = Path(data_dir)
    podcast_dirs = [p.parent for p in data_path.rglob("*.json")]
    logging.info(f"Found {len(list(set(podcast_dirs)))} podcast directories in {data_dir}")
    return sorted(list(set(podcast_dirs)))


def metadata_generator(podcast_dirs):
    """
    A fast generator that yields only the metadata (paths and text),
    not the actual audio data.
    """
    for podcast_dir in podcast_dirs:
        m4a_files = sorted(podcast_dir.glob("*.m4a"), key=natural_sort_key)
        json_files = list(podcast_dir.glob("*.json"))

        if not json_files:
            logging.warning(f"Skipping directory {podcast_dir}: No JSON file found.")
            continue

        json_data = json.load(open(json_files[0], 'r', encoding='utf-8'))
        if not json_data:
            logging.warning(f"Skipping {podcast_dir.name}: JSON file is empty.")
            continue

        if isinstance(json_data[0], dict):
            texts = [item['text'] for item in json_data]
        else:
            logging.warning(f"Skipping {podcast_dir.name}: Unknown JSON structure in {json_files[0]}.")
            continue

        if len(m4a_files) != len(texts):
            logging.warning(
                f"Skipping directory {podcast_dir}: Mismatch between audio files ({len(m4a_files)}) and texts ({len(texts)}).")
            continue

        podcast_id = podcast_dir.name
        clips = []
        for i, (audio_fp, text) in enumerate(zip(m4a_files, texts)):
            clips.append({
                "podcast_id": podcast_id,
                "text": text,
                "audio_path": str(audio_fp), # Store path as string
                "clip_id": i
            })
        yield {"podcast_id": podcast_id, "processed_clips": clips}


def load_and_encode_audio(item):
    """
    The mapping function that will be applied in parallel.
    It takes one example (dict), loads the audio, encodes it,
    and returns the updated example.
    """
    item["raw_audio"] = encode_audio_to_base64(Path(item["audio_path"]))
    tag = TinyTag.get(item["audio_path"])
    item["audio_duration"] = tag.duration
    item["audio_sr"] = tag.samplerate
    return item


def load_podcast(example):
    for i, item in enumerate(example['processed_clips']):
        example['processed_clips'][i] = load_and_encode_audio(item)
    return example


def main():
    parser = argparse.ArgumentParser(description="Build a pre-tokenized audio dataset for HiggsAudio finetuning.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the raw podcast data.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the processed dataset will be saved.")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Number of processes to use for audio encoding.")
    # --- New arguments added here ---
    parser.add_argument("--test_samples", type=int, default=0,
                        help="Number of samples to hold out for the test set. If 0, no split is performed.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for shuffling the dataset before splitting.")
    parser.add_argument("--keep_cache", action="store_true",
                        help="Keep cache files after processing. If not set, cache will be deleted to save disk space.")
    args = parser.parse_args()

    logging.info(f"Starting dataset creation with args: {args}")

    podcast_dirs = find_podcasts(args.data_dir)
    if not podcast_dirs:
        logging.error(f"No podcast data found in {args.data_dir}. Exiting.")
        return

    # Step 1: Create a lightweight dataset with metadata only
    logging.info("Creating initial metadata dataset...")
    metadata_dataset = Dataset.from_generator(
        metadata_generator,
        gen_kwargs={"podcast_dirs": podcast_dirs},
    )
    logging.info(f"Metadata dataset created with {len(metadata_dataset)} samples.")

    # Step 2: Prepare output directory and cache
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Use .map() to process the audio in parallel
    # Use memory-optimized settings to prevent OOM and PyArrow 2GB limit
    logging.info(f"Processing audio files with {args.num_proc} processes...")
    final_dataset = metadata_dataset.map(
        load_podcast,
        num_proc=args.num_proc,
        keep_in_memory=False,  # Don't keep entire dataset in memory
        cache_file_name=str(cache_dir / "encoded_audio.arrow"),  # Use disk cache
        writer_batch_size=1,  # Write one example at a time to avoid PyArrow 2GB limit
        desc="Encoding audio files",
    )

    # --- Step 4: Shuffle and split the dataset if requested ---

    # Check if a split is requested and possible
    if 0 < args.test_samples < len(final_dataset):
        logging.info(f"Shuffling and splitting dataset into train and test sets ({args.test_samples} test samples)...")

        # 1. Shuffle the dataset first for a random split. Use indices to reduce memory usage.
        shuffled_dataset = final_dataset.shuffle(
            seed=args.random_seed,
            keep_in_memory=False,  # Don't load entire dataset in memory during shuffle
        )

        # 2. Split the dataset. This is also memory-safe.
        # It returns a DatasetDict: {'train': Dataset, 'test': Dataset}
        split_dataset = shuffled_dataset.train_test_split(test_size=args.test_samples)

        # 3. Save the DatasetDict. This will automatically create 'train' and 'test' subdirectories.
        train_path = output_path / "train"
        test_path = output_path / "test"

        logging.info(f"Saving training set ({len(split_dataset['train'])} samples) to {train_path}")
        logging.info(f"Saving test set ({len(split_dataset['test'])} samples) to {test_path}")

        split_dataset['train'].save_to_disk(str(train_path))
        split_dataset['test'].save_to_disk(str(test_path))

    else:
        # If no split is needed, save the entire dataset as before.
        if args.test_samples > 0:
            logging.warning(
                f"Requested test_samples ({args.test_samples}) is invalid for dataset of size {len(final_dataset)}. Saving as a single dataset.")

        logging.info(f"Saving single dataset ({len(final_dataset)} samples) to {output_path}...")
        final_dataset.save_to_disk(str(output_path))

    logging.info("Dataset creation complete!")
    
    # Clean up cache if requested
    if not args.keep_cache and cache_dir.exists():
        import shutil
        logging.info(f"Cleaning up cache directory: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            logging.info(f"Cache directory deleted, freed up disk space")
        except Exception as e:
            logging.warning(f"Failed to delete cache directory: {e}")

if __name__ == "__main__":
    main()
