import os
from pathlib import Path
import folder_paths
import shutil
from .inference_core import HSyn9DHPEInference


class HSyn9DHPENode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "smooth_detections": ("BOOLEAN", {"default": False}),
                "smooth_3d": ("BOOLEAN", {"default": False}),
                "open_houdini": ("BOOLEAN", {"default": True}),
                "use_hython": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("keypoints_render_dir", "body_render_dir", "start_image_path", "total_frames")
    FUNCTION = "run_inference"
    CATEGORY = "pose/3d"
    OUTPUT_NODE = True
    
    def run_inference(self, video, smooth_detections=False, smooth_3d=False, 
                     open_houdini=True, use_hython=True):

        node_dir = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.join(node_dir, "HSyn9DHPE")
        python_path = os.path.join(repo_path, "env", "python.exe")
        checkpoint_path = os.path.join(repo_path, "checkpoints", "model.pth")
        houdini_scene = os.path.join(node_dir, "houdini", "houdini_render.hiplc")
        
        base_output_dir = folder_paths.get_output_directory()
        output_dir = os.path.join(base_output_dir, "HSyn9DHPE")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        video_path = video._VideoFromFile__file
        
        inference = HSyn9DHPEInference(
            python_path=python_path,
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            houdini_scene=houdini_scene
        )
        
        results = inference.run_inference(
            video_path=video_path,
            output_dir=output_dir,
            smooth_detections=smooth_detections,
            smooth_3d=smooth_3d,
            open_houdini=open_houdini,
            use_hython=use_hython
        )
        
        total_frames = results['num_frames']
        
        keypoints_render_dir = os.path.join(output_dir, "keypoints_render")
        body_render_dir = os.path.join(output_dir, "body_render")
        start_image_path = os.path.join(output_dir, "start_image.jpg")
        
        os.makedirs(keypoints_render_dir, exist_ok=True)
        os.makedirs(body_render_dir, exist_ok=True)

        return (keypoints_render_dir, body_render_dir, start_image_path, total_frames)


NODE_CLASS_MAPPINGS = {
    "HSyn9DHPE": HSyn9DHPENode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSyn9DHPE": "HSyn9DHPE - 9D Pose Estimation"
}