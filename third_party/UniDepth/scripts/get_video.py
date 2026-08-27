import cv2
import os
from pathlib import Path

def frames_to_video(input_folder, output_video, fps=30):
    """
    Convert a sequence of image frames to a video.
    
    Args:
        input_folder (str): Path to folder containing image frames
        output_video (str): Path for output video file
        fps (int): Frames per second for the output video
    """
    # Get list of image files
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    # import pdb;pdb.set_trace()
    if not images:
        print("No images found in the specified folder!")
        return
    
    # Read first image to get dimensions
    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_frame.shape
    # import pdb;pdb.set_trace()
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write each frame to video
    for image in images:
        frame = cv2.imread(os.path.join(input_folder, image))
        video_writer.write(frame)
    
    # Release video writer          
    video_writer.release()
    print(f"Video saved as {output_video}")

# Example usage
if __name__ == "__main__":
    input_folder = "./datasets/testdata/zoomin_out/"  # Replace with your frames folder path
    output_video = "./datasets/testdata/zoomin_out.mp4"      # Replace with desired output path
    fps = 11                              # Frames per second
    frames_to_video(input_folder, output_video, fps)