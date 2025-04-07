import os
import glob
import subprocess
import ffmpeg

def extract_frames(video_path, output_dir, fps=1, num_frames=None):
    """
    Extract frames from a video using ffmpeg.

    Parameters:
    - video_path: path to the video file
    - output_dir: directory to save the extracted frames
    - fps: frames per second to extract (used if num_frames is None or invalid)
    - num_frames: the exact number of frames to extract (optional)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clean old frame files
    for frame_file in glob.glob(os.path.join(output_dir, "frame_*.jpg")):
        os.remove(frame_file)

    # Get video duration
    try:
        probe = ffmpeg.probe(video_path, v="error", select_streams="v:0", show_entries="format=duration")
        video_duration = float(probe["format"]["duration"])
    except Exception as e:
        raise RuntimeError(f"Error getting video duration: {e}")

    # Validate num_frames
    if num_frames and num_frames > video_duration:
        print(f"[!] Requested {num_frames} frames, but video is only {video_duration:.2f} seconds long.")
        print(f"[→] Falling back to fps={fps} mode.")
        num_frames = None

    if num_frames:
        interval = video_duration / num_frames
        frame_timestamps = [interval * i for i in range(num_frames)]

        for i, timestamp in enumerate(frame_timestamps):
            cmd = [
                "ffmpeg",
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                os.path.join(output_dir, f"frame_{i+1:04d}.jpg")
            ]
            subprocess.run(cmd, check=True)
    else:
        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            output_pattern
        ]
        subprocess.run(cmd, check=True)

#Example Usage
if __name__ == "__main__":
    test_video = "friends_phonebill.mp4"
    output_folder = "frames"
    frames_to_extract = 5

    print(f"[TEST] Extracting {frames_to_extract} frames from '{test_video}' into '{output_folder}'...")
    extract_frames(test_video, output_folder, num_frames=frames_to_extract)
    print("[DONE] Frame extraction complete.")
