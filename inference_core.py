import os
import subprocess
import json
from pathlib import Path


class HSyn9DHPEInference:
    def __init__(self, python_path: str, repo_path: str, checkpoint_path: str, houdini_scene: str):
        self.python_path = python_path
        self.repo_path = Path(repo_path)
        self.checkpoint_path = checkpoint_path
        self.houdini_scene = houdini_scene
        self.inference_script = self.repo_path / "src" / "inference.py"
        self.houdini_exe = r"C:\Program Files\Side Effects Software\Houdini 20.5.445\bin\houdini.exe"
        self.hython_exe = r"C:\Program Files\Side Effects Software\Houdini 20.5.445\bin\hython.exe"
    
    def run_inference(self, video_path: str, output_dir: str, 
                     smooth_detections: bool = False, smooth_3d: bool = False,
                     open_houdini: bool = True, use_hython: bool = True):
        
        cmd = [
            self.python_path,
            str(self.inference_script),
            video_path,
            output_dir,
            "--checkpoint", self.checkpoint_path
        ]
        
        if smooth_detections:
            cmd.append("--smooth_detections")
        
        if smooth_3d:
            cmd.append("--smooth_3d")
        
        subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.repo_path))

        results_json = Path(output_dir) / "results_frame.json"
        with open(results_json, 'r') as f:
            results_data = json.load(f)
        
        num_frames = len(results_data)
        
        if open_houdini:
            if use_hython:
                self._render_with_hython(num_frames, output_dir)
            else:
                self._open_houdini_gui(num_frames, output_dir)
        
        return {
            'results_json_path': str(results_json),
            'houdini_frames_dir': str(Path(output_dir) / "houdini_frames"),
            'output_dir': output_dir,
            'results_data': results_data,
            'num_frames': num_frames
        }
    
    def _render_with_hython(self, sequence_length: int, output_dir: str):
        
        scene_path = str(self.houdini_scene).replace('\\', '/')
        
        hython_script = f"""import hou
hou.hipFile.load("{scene_path}")
control_node = hou.node("/obj/CONTROLLER")
control_node.parm("end_frame").set({sequence_length - 1})
control_node.parm("render").pressButton()
"""
        
        script_path = Path(output_dir) / "hython_render.py"
        with open(script_path, 'w') as f:
            f.write(hython_script)
        
        result = subprocess.run(
            [self.hython_exe, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(output_dir)
        )
        
        print(f"[Hython] Exit code: {result.returncode}")
        if result.stdout:
            print(f"[Hython] Output: {result.stdout}")
        if result.stderr:
            print(f"[Hython] Errors: {result.stderr}")
        
        if result.returncode != 0:
            raise RuntimeError(f"Hython failed with exit code {result.returncode}")
        
        return
    
    def _open_houdini_gui(self, sequence_length: int, output_dir: str):
        
        houdini_script = f"""import hou
control_node = hou.node("/obj/CONTROLLER")
control_node.parm("end_frame").set({sequence_length - 1})
"""

        script_path = Path(output_dir) / "houdini_gui_render.py"
        with open(script_path, 'w') as f:
            f.write(houdini_script)
        
        subprocess.Popen(
            [self.houdini_exe, self.houdini_scene, str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(output_dir)
        )
        
        return