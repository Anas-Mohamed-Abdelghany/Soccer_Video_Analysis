import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

@dataclass
class VideoConfig:
    width: int
    height: int
    fps: int

class ObjectTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.player_positions: Dict[int, List[Tuple[int, int]]] = {}
        self.video_config: Optional[VideoConfig] = None

    def set_video_config(self, width: int, height: int, fps: int) -> None:
        self.video_config = VideoConfig(width=width, height=height, fps=fps)

    def _process_batch(self, frames: List[np.ndarray], batch_size: int = 20) -> List:
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections.extend(batch)
        return detections

    def _update_player_position(self, track_id: int, bbox: List[float]) -> None:
        if track_id not in self.player_positions:
            self.player_positions[track_id] = []
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 2)
        self.player_positions[track_id].append((x_center, y_center))

    def get_tracks(self, frames: List[np.ndarray]) -> Dict[str, List[Dict]]:
        detections = self._process_batch(frames)
        tracks = {category: [dict() for _ in frames] for category in ["players", "referees", "ball"]}
        
        for frame_num, detection in enumerate(detections):
            cls_map = {v: k for k, v in detection.names.items()}
            det_sv = sv.Detections.from_ultralytics(detection)
            
            # Convert goalkeepers to players
            goalkeeper_indices = det_sv.class_id == cls_map["goalkeeper"]
            det_sv.class_id[goalkeeper_indices] = cls_map["player"]
            
            tracked_det = self.tracker.update_with_detections(det_sv)
            
            for det in tracked_det:
                bbox, cls_id, track_id = det[0].tolist(), det[3], det[4]
                
                if cls_id == cls_map['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    self._update_player_position(track_id, bbox)
                elif cls_id == cls_map['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    
            # Handle ball detection separately as it doesn't need tracking
            for det in det_sv:
                if det[3] == cls_map['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": det[0].tolist()}
                    
        return tracks

    def draw_heatmap(self, field_image: np.ndarray, player_id: int) -> np.ndarray:
        if player_id not in self.player_positions or not self.video_config:
            return field_image
            
        height, width = field_image.shape[:2]
        heatmap = np.zeros_like(field_image[:, :, 0], dtype=np.float32)
        
        for x, y in self.player_positions[player_id]:
            x_scaled = int((x / self.video_config.width) * width)
            y_scaled = int((y / self.video_config.height) * height)
            
            if 0 <= x_scaled < width and 0 <= y_scaled < height:
                kernel_size = int(min(width, height) * 0.05)
                cv2.circle(heatmap, (x_scaled, y_scaled), kernel_size, 1, -1)
        
        blur_radius = int(min(width, height) * 0.1)
        blur_radius += not blur_radius % 2  # Ensure odd kernel size
        heatmap = cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        
        heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(field_image, 0.4, heatmap_colored, 0.6, 0)

class VideoRenderer:
    @staticmethod
    def draw_annotations(frames: List[np.ndarray], tracks: Dict) -> List[np.ndarray]:
        return [VideoRenderer._annotate_frame(frame.copy(), frame_tracks) 
                for frame, frame_tracks in zip(frames, zip(tracks["players"], tracks["referees"], tracks["ball"]))]

    @staticmethod
    def _annotate_frame(frame: np.ndarray, frame_tracks: Tuple[Dict, Dict, Dict]) -> np.ndarray:
        players, referees, balls = frame_tracks
        
        for track_id, player in players.items():
            frame = VideoRenderer._draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
            
        for _, referee in referees.items():
            frame = VideoRenderer._draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
        for _, ball in balls.items():
            frame = VideoRenderer._draw_triangle(frame, ball["bbox"], (0, 255, 0))
            
        return frame

    @staticmethod
    def _draw_ellipse(frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int], 
                      track_id: Optional[int] = None) -> np.ndarray:
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        width = int(bbox[2] - bbox[0])
        
        cv2.ellipse(frame, (center_x, center_y), (width // 2, int(0.35 * width)),
                    0.0, -45, 235, color, 2, cv2.LINE_4)
                    
        if track_id is not None:
            cv2.putText(frame, str(track_id), (center_x, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    @staticmethod
    def _draw_triangle(frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int]) -> np.ndarray:
        x = int((bbox[0] + bbox[2]) / 2)
        y = int(bbox[1])
        
        points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
        return frame

class VideoAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Soccer Video Analysis")
        self.root.geometry("800x600")
        
        self.paths = {"video": "", "field": "", "output": "output_video.avi"}
        self.tracker = ObjectTracker("models/yolov5s.pt")
        self.frames = []
        self.tracks = None
        self.player_ids = {}
        
        self._setup_gui()
        
    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self._create_file_section(main_frame)
        self._create_processing_section(main_frame)
        self._create_player_section(main_frame)
        self._create_visualization_section(main_frame)
        self._create_status_section(main_frame)
        
    def _create_file_section(self, parent):
        frame = self._create_section(parent, "File Selection", 0)
        ttk.Button(frame, text="Select Video", command=self._select_video).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Select Field Image", command=self._select_field).grid(row=0, column=1, padx=5, pady=5)
        
    def _create_processing_section(self, parent):
        frame = self._create_section(parent, "Processing", 1)
        ttk.Button(frame, text="Process Video", command=self._process_video).grid(row=0, column=0, padx=5, pady=5)
        
    def _create_player_section(self, parent):
        frame = self._create_section(parent, "Player Selection", 2)
        self.player_combo = ttk.Combobox(frame, state="readonly")
        self.player_combo.grid(row=0, column=0, padx=5, pady=5)
        
    def _create_visualization_section(self, parent):
        frame = self._create_section(parent, "Visualization", 3)
        ttk.Button(frame, text="Show Heatmap", command=self._show_heatmap).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Play Video", command=self._play_video).grid(row=0, column=1, padx=5, pady=5)
        
    def _create_status_section(self, parent):
        frame = self._create_section(parent, "Status", 4)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status_var).grid(row=0, column=0, padx=5, pady=5)
        
    @staticmethod
    def _create_section(parent, text, row):
        frame = ttk.LabelFrame(parent, text=text, padding="5")
        frame.grid(row=row, column=0, sticky="ew", pady=5)
        return frame
        
    def _select_video(self):
        self.paths["video"] = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        )
        self._update_status(f"Selected Video: {os.path.basename(self.paths['video'])}")
        
    def _select_field(self):
        self.paths["field"] = filedialog.askopenfilename(
            title="Select Field Image",
            filetypes=(("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*"))
        )
        self._update_status(f"Selected Field Image: {os.path.basename(self.paths['field'])}")
        
    def _process_video(self):
        if not all([self.paths["video"], self.paths["field"]]):
            messagebox.showerror("File Error", "Please select both a video and a field image.")
            return
            
        cap = cv2.VideoCapture(self.paths["video"])
        self.tracker.set_video_config(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(cap.get(cv2.CAP_PROP_FPS))
        )
        
        self.frames = self._load_video_frames(cap)
        cap.release()
        
        if self.frames:
            self.tracks = self.tracker.get_tracks(self.frames)
            self._save_processed_video()
            self._update_player_list()
            self._update_status("Processing Complete")
            
    def _load_video_frames(self, cap):
        if not cap.isOpened():
            messagebox.showerror("Video Error", "Could not open the video file.")
            return []
            
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames
        
    def _save_processed_video(self):
        if not self.frames:
            return
            
        out = cv2.VideoWriter(
            self.paths["output"],
            cv2.VideoWriter_fourcc(*'XVID'),
            self.tracker.video_config.fps,
            (self.frames[0].shape[1], self.frames[0].shape[0])
        )
        
        annotated_frames = VideoRenderer.draw_annotations(self.frames, self.tracks)
        for frame in annotated_frames:
            out.write(frame)
        out.release()
        
    def _update_player_list(self):
        unique_players = sorted(set(
            player_id for frame in self.tracks["players"]
            for player_id in frame.keys()
        ))
        self.player_ids = {str(pid): pid for pid in unique_players}
        self.player_combo['values'] = list(self.player_ids.keys())
        if self.player_combo['values']:
            self.player_combo.set(self.player_combo['values'][0])
            
    def _show_heatmap(self):
        if not all([self.paths["field"], self.tracks]):
            messagebox.showerror("Error", "Please process video first.")
            return
            
        selected = self.player_combo.get()
        if selected:
            field_image = cv2.imread(self.paths["field"])
            heatmap = self.tracker.draw_heatmap(field_image, self.player_ids[selected])
            cv2.imshow(f"Heatmap for Player {selected}", heatmap)
            
    def _play_video(self):
        cap = cv2.VideoCapture(self.paths["output"])
        if not cap.isOpened():
            messagebox.showerror("Video Error", "Could not open the processed video.")
            return
            
        frame_time = int(1000 / self.tracker.video_config.fps)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.imshow("Processed Video", frame)
            if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def _update_status(self, message: str):
        self.status_var.set(message)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    VideoAnalyzerGUI().run()
