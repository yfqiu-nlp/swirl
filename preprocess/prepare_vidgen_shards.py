import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VidGenSharding")

def create_shards(base_path: Path, output_dir: Path, num_shards: int, shard_size: int):
    """
    Scans VidGen-1M for frame pairs and creates fixed metadata shards.
    """
    logger.info(f"Scanning {base_path} for valid samples...")

    valid_samples = []

    if not base_path.exists():
        logger.error(f"Base path not found: {base_path}")
        return

    split_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    for split_dir in tqdm(split_dirs, desc="Scanning Splits"):
        video_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for v_dir in video_dirs:
            f0 = v_dir / "frame_0.png"
            f1 = v_dir / "frame_1.png"

            if f0.exists() and f1.exists():
                valid_samples.append(str(v_dir.relative_to(base_path)))

    total_found = len(valid_samples)
    required_samples = num_shards * shard_size

    logger.info(f"Found {total_found} valid samples. Required for full shards: {required_samples}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size

        if start_idx >= len(valid_samples):
            logger.warning(f"Not enough data for shard {i}. Stopping.")
            break

        shard_data = valid_samples[start_idx:end_idx]

        output_file = output_dir / f"meta_shard_{i}.json"

        with open(output_file, 'w') as f:
            json.dump(shard_data, f, indent=2)

        logger.info(f"Created {output_file} | Samples: {len(shard_data)}")

    logger.info("Sharding complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VidGen-1M Shards")
    parser.add_argument('--base_path', type=str, required=True, help="Path to VIDGEN-1M_images root")
    parser.add_argument('--output_dir', type=str, default="./vidgen_shards", help="Where to save meta_shard_X.json files")
    parser.add_argument('--num_shards', type=int, default=5, help="Number of shards to create")
    parser.add_argument('--shard_size', type=int, default=50000, help="Number of samples per shard")

    args = parser.parse_args()

    create_shards(
        Path(args.base_path),
        Path(args.output_dir),
        args.num_shards,
        args.shard_size
    )
