import argparse
import os
import random

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def compute_ssim(img1, img2):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Parameters:
    - img1: First image (NumPy array, RGB).
    - img2: Second image (NumPy array, RGB).

    Returns:
    - float: The SSIM value.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(gray1, gray2, full=True)[0]


def resize_to_l(image, l=512):
    """
    Resize the shorter edge of an image to length 'l', maintaining aspect ratio.
    """
    height, width = image.shape[:2]

    if height < width:
        new_height = l
        new_width = int((new_height / height) * width)
    else:
        new_width = l
        new_height = int((new_width / width) * height)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def read_random_frames(filename, interval_seconds=5, num_frames=2, retry=5):
    """
    Randomly sample frames from a video at a given interval and return if they are dissimilar enough.
    Replaces Decord with cv2.VideoCapture.

    Parameters:
    - filename: Path to the input video file.
    - interval_seconds: Time interval (in seconds) between sampled frames.
    - num_frames: Number of frames to sample.
    - retry: Number of retries if sampling fails or frames are too similar.

    Returns:
    - A list of sampled frames (as NumPy arrays in RGB format), or None if retries are exhausted.
    """
    exception = None
    for _ in range(retry):
        cap = None
        try:
            cap = cv2.VideoCapture(filename)
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {filename}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0 or fps <= 0:
                 raise ValueError(f"Invalid video properties: frames={total_frames}, fps={fps}")

            frame_interval = int(fps * interval_seconds)

            min_required_frames = frame_interval * (num_frames - 1) + 1

            if total_frames < min_required_frames:
                raise ValueError(f"Video is too short ({total_frames} frames) for the required sampling ({min_required_frames} frames needed).")

            valid_end = total_frames - min_required_frames
            start_frame = random.randint(0, valid_end)

            frames = []
            for i in range(num_frames):
                frame_idx = start_frame + i * frame_interval

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()

                if not ret or frame is None:
                    raise IOError(f"Failed to read frame at index {frame_idx} from {filename}")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb.astype(np.uint8))

            frame1, frame2 = frames[0], frames[1]
            frame1_resized, frame2_resized = resize_to_l(frame1, 256), resize_to_l(frame2, 256)
            similarity = compute_ssim(frame1_resized, frame2_resized)

            if similarity < 0.95:
                return frames

        except Exception:
            continue

        finally:
            if cap:
                cap.release()

    if exception is not None:
         print(f"⚠️ All retries failed for {filename}. Last error: {exception}")

    return None


def split_list(lst, N, K):
    """
    Split a list into N nearly equal parts and return the K-th part (1-based index).
    """
    if N <= 0 or K <= 0 or K > N:
        raise ValueError("N must be > 0 and K must be between 1 and N")

    length = len(lst)
    chunk_size = length // N
    remainder = length % N

    chunks = []
    start = 0
    for i in range(N):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks[K - 1]


def calculate_low_pixel_ratio(image: np.ndarray, threshold: int = 5) -> float:
    """
    Calculate the ratio of pixels whose values in all 3 channels are below the given threshold.
    Useful for removing videos with black borders.

    Parameters:
        image (np.ndarray): Input image of shape (H, W, C), expected RGB.
        threshold (int): Pixel value threshold (default: 5).

    Returns:
        float: Proportion of pixels below the threshold across all 3 channels.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3)")

    total_pixels = image.shape[0] * image.shape[1]
    low_pixel_mask = np.all(image < threshold, axis=2)
    low_pixel_count = np.sum(low_pixel_mask)

    return low_pixel_count / total_pixels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Frame sampling and filtering pipeline.")
    parser.add_argument("--storage_path", type=str, required=True, help="Dataset storage path.")
    parser.add_argument("-k", type=int, required=True, help="Index of the chunk to process")
    parser.add_argument("-N", type=int, required=True, help="Total number of chunks")
    args = parser.parse_args()
    k, N = args.k, args.N

    total_index = [i for i in range(1, 2048)]
    part_indexes = split_list(total_index, N, k)


    save_root = f'{args.storage_path}/VIDGEN-1M_images'                
    data_dir = f'{args.storage_path}/VIDGEN-1M_videos'                
    interval_seconds = 6                        

    try:
        print(f"🚀 Starting chunk {k} of {N}, processing indexes: {part_indexes[0]}...{part_indexes[-1]}")
        for i in tqdm(part_indexes, desc=f"Overall Chunk {k}"):
            folder_name = f'VidGen_video_{i}'
            save_dir = os.path.join(save_root, folder_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print('Created new folder:', save_dir)

            source_dir = os.path.join(data_dir, folder_name)

            if not os.path.exists(source_dir):
                print(f"⚠️ Source directory not found: {source_dir}. Skipping.")
                continue

            video_names = os.listdir(source_dir)

            print(f'Number of videos in {folder_name}: {len(video_names)}')

            for video_name in tqdm(video_names, leave=False, desc="Processing videos"):
                try:
                    video_path = os.path.join(source_dir, video_name)
                    save_path = os.path.join(save_dir, f"{video_name.split('.')[0]}")

                    if not os.path.isfile(video_path):
                        print(f"File not found: {video_path}. Skipping.")
                        continue

                    frame_lst = read_random_frames(video_path, interval_seconds)

                    if frame_lst is not None:

                        frame_0, frame_1 = frame_lst[0], frame_lst[1]

                        ratio1 = calculate_low_pixel_ratio(frame_0)
                        ratio2 = calculate_low_pixel_ratio(frame_1)

                        if ratio1 < 0.1 and ratio2 < 0.1:

                            frame_0_bgr = frame_0[:, :, ::-1]
                            frame_1_bgr = frame_1[:, :, ::-1]

                            if not os.path.exists(save_path):
                                os.makedirs(save_path)

                            cv2.imwrite(os.path.join(save_path, "frame_0.png"), frame_0_bgr)
                            cv2.imwrite(os.path.join(save_path, "frame_1.png"), frame_1_bgr)

                            concat_frame = cv2.hconcat(frame_lst)
                            save_name = os.path.join(save_path, "concat_frame.png")

                            cv2.imwrite(save_name, concat_frame[:, :, ::-1])

                except KeyboardInterrupt:
                    print("\n⛔ Interrupted by user. Exiting cleanly.")
                    raise                                
                except Exception:
                    print(f"⚠️ Error processing video {video_name}")

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user. Program terminated.")
    except Exception:
        print("===== Unexpected error occurred =====")
        import traceback
        traceback.print_exc()
