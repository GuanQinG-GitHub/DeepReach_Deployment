import cv2
import os
import glob
import re

def create_video_from_images(image_folder, video_name, fps=30):
    images = glob.glob(os.path.join(image_folder, "BRS_validation_plot_epoch_*.png"))
    
    # Sort images by epoch number
    # Extract number from filename "BRS_validation_plot_epoch_1000.png"
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0

    images.sort(key=extract_epoch)

    if not images:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(images)} images.")
    
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    # mp4v is a good option for .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        print(f"Writing {image}...")
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved as {video_name}")

if __name__ == "__main__":
    # Path provided by user
    # C:\Users\Leixi\Desktop\DeepReach\deepreach\runs\dubins3dDiscounted_trial_2\training\checkpoints
    
    # Use relative path if possible, or absolute
    # Assuming script is run from deepreach root
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_folder = os.path.join(base_dir, 'runs', 'dubins3dDiscounted_trial_dec_3', 'training', 'checkpoints')
    
    # User requested 0.01s per frame -> 100 FPS
    fps = 10 
    
    video_name = 'training_evolution_dec_3.mp4'
    
    if os.path.exists(image_folder):
        create_video_from_images(image_folder, video_name, fps)
    else:
        print(f"Directory not found: {image_folder}")
        # Fallback to absolute path provided by user if relative fails
        abs_path = r"/share/dlee/xzhan245/projects/DeepReach_Deployment/runs/dubins3dDiscounted_trial_dec_3\training\checkpoints"
        if os.path.exists(abs_path):
             create_video_from_images(abs_path, video_name, fps)
        else:
            print("Could not locate image folder.")
