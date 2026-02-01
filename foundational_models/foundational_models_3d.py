"""
Foundation Models for 3D Semantic Scene Understanding
Author: Oluoma

This project integrates modern vision foundation models
to perform semantic segmentation and monocular depth estimation
from a single RGB image.

Research Foundations:
- Depth Anything V2:
  https://arxiv.org/pdf/2406.09414

- Segment Anything Model 2 (SAM 2):
  https://arxiv.org/abs/2408.00714

This is an engineering and systems-level implementation
built on top of publicly released research and checkpoints.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
from typing import List, Tuple, Optional
import urllib.request
import sys
sys.path.append("./Depth-Anything-V2")



print("="*70)
print("FOUNDATIONAL MODELS FOR SEMANTIC SEGMENTATION")
print("SAM 2 + DEPTH ANYTHING V2")
print("="*70)

# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_PATH = "images/test_image.jpg"

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'\n Using device: {DEVICE}')
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
print("CUDA version:", torch.version.cuda)


# ==================================================
# DEPTH ANYTHING V2 - State-of-the-Art Depth
# ==================================================

class DepthAnythingV2:
    """
    Depth Anything V2
    Research Paper : https://arxiv.org/pdf/2406.09414
    """
    def __init__(self, model_size="small"):
        print(f"\n[1/2] Loading Depth Anything V2 ({model_size})...")
        self.available = False

        try:
            import sys, os, torch
            sys.path.append("Depth-Anything-V2")

            from depth_anything_v2.dpt import DepthAnythingV2 as DAv2Model

            # Model configs
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }

            # Map user-friendly names
            encoder_map = {
                "small": "vits",
                "base": "vitb",
                "large": "vitl"
            }

            encoder = encoder_map.get(model_size, "vits")

            # Correct checkpoint path
            checkpoint_path = f"checkpoints/depth_anything_v2_{encoder}.pth"

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

            # Load model
            self.model = DAv2Model(**model_configs[encoder])
            state = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(state)

            self.model.to(DEVICE)
            self.model.eval()

            self.available = True
            self.encoder = encoder

            print(f"Depth Anything V2 ({encoder}) loaded successfully")

        except Exception as e:
            print(f"Could not load Depth Anything V2: {e}")
            print("Falling back to alternative depth estimation...")
            self.available = False
            self._load_fallback()

    def _load_fallback(self):
        """
        Load MiDaS as fallback Mixed Depth Scale)
        MiDaS created the pioneering work on which Depth Anything, Metric3D and ZeroDepth was created.
        """
        try:
            print("Loading MiDaS v3.1 as a fallback ...")
            self.model = torch.hub.load('intel-isl/MiDaS','DPT_Hybrid',pretrained=True)
            self.model.to(DEVICE)
            self.model.eval()

            midas_transforms = torch.hub.load('intel-isl/MiDaS','transforms')
            self.transform = midas_transforms.dpt_transform
            self.available = True
            print("MiDaS loaded as a fallback ...")
        except Exception as e:
            print(f"Could not load MiDaS: {e}")
            self.available = False

    def predict(self, image : np.ndarray) -> np.ndarray:
        """
        Predict Depth map from Image
        """
        if not self.available:
            return self._simple_depth(image)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare input
        h, w = rgb.shape[:2]

        if self.encoder in ['vits', 'vitb', 'vitl']:
            # Depth Anything V2 - Uses infer_image method directly
            depth = self.model.infer_image(rgb)  # Returns numpy array
            depth_map = depth.astype(np.float32)
            
        elif self.encoder == 'midas':
            # MiDaS preprocessing
            input_batch = self.transform(rgb).to(DEVICE)
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
        else:
            return self._simple_depth(image)

        # Normalize to 0-1 (invert so closer = higher value)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        depth_map = 1.0 - depth_map  # invert: 1 = close, 0 = far

        return depth_map

    def _simple_depth(self,image):
        "Fallback simple depth estimation"
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        depth = cv2.GaussianBlur(255 - edges, (21,21),0)
        depth = depth.astype(float) / 255.0
        return depth

# ==================================================
# SAM 2 - Segmen Anything Model 2
# ==================================================
class RealSAM2:
    """
    SAM 2 - Segment Anything Model 2 (Meta, 2024)
    Research Paper : https://arxiv.org/abs/2408.00714
    """
    
    def __init__(self, checkpoint_path=None):
        import os
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        print("\n[2/2] Loading SAM...")

        self.sam_available = False

        try:
            print("Attempting to load SAM model...")

            # Default checkpoint
            checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"

            # Check checkpoint exists
            if not os.path.exists(checkpoint_path):
                print(f"SAM checkpoint not found at: {checkpoint_path}")
                print("Download from:")
                print(checkpoint_url)
                print("\nOr use one of these smaller models:")
                print("- sam_vit_l_0b3195.pth (Large)")
                print("- sam_vit_b_01ec64.pth (Base)")
                raise FileNotFoundError("SAM checkpoint not found")

            # Device
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            # Load SAM model
            model_type = "vit_h"  # or vit_l, vit_b
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(DEVICE)

            # Store model
            self.sam = sam

            # Create mask generator
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )

            self.sam_available = True
            print("SAM loaded successfully!")
            print(f"  Model: {model_type}")
            print(f"  Device: {DEVICE}")

        except Exception as e:
            print(f"Could not load SAM: {e}")
            print("Falling back to traditional segmentation")
            self.sam_available = False

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Segment image using SAM or fallback
        """
        if self.sam_available:
            return self._segment_with_sam(image)
        else:
            return self._segment_fallback(image)
    
    def _segment_with_sam(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Use actual SAM for segmentation"""
        print("Using SAM for segmentation...")
        
        # SAM expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = self.mask_generator.generate(rgb)
        
        print(f"SAM found {len(masks)} segments")
        
        # Convert SAM masks to our format
        h, w = image.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.int32)
        
        # Sort masks by area (largest first)
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Assign each mask a unique ID
        segments_info = []
        for idx, mask_dict in enumerate(masks_sorted):
            mask = mask_dict['segmentation']
            seg_mask[mask] = idx
            
            # Extract info
            bbox = mask_dict['bbox']  # [x, y, w, h]
            segments_info.append({
                'id': idx,
                'area': int(mask_dict['area']),
                'bbox': [int(bbox[0]), int(bbox[1]), 
                        int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])],
                'stability_score': float(mask_dict['stability_score']),
                'predicted_iou': float(mask_dict['predicted_iou'])
            })
        
        return seg_mask, segments_info
    
    def _segment_fallback(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Fallback segmentation using SLIC + K-means"""
        print("Using fallback segmentation (SLIC + K-means)...")
        
        from skimage.segmentation import slic
        from skimage.color import rgb2lab
        from sklearn.cluster import KMeans
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lab = rgb2lab(rgb)
        
        # SLIC superpixels
        num_segments = 15
        segments = slic(rgb, n_segments=num_segments*10, compactness=10, sigma=1)
        
        # Extract features
        num_superpixels = segments.max() + 1
        features = []
        
        for i in range(num_superpixels):
            mask_i = segments == i
            if mask_i.sum() > 0:
                avg_lab = lab[mask_i].mean(axis=0)
                positions = np.argwhere(mask_i)
                avg_pos = positions.mean(axis=0)
                features.append(np.concatenate([avg_lab, avg_pos]))
        
        features = np.array(features)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(num_segments, len(features)), random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Create mask
        mask = np.zeros_like(segments)
        for i in range(num_superpixels):
            if i < len(cluster_labels):
                mask[segments == i] = cluster_labels[i]
        
        # Create segment info
        segments_info = []
        for seg_id in range(num_segments):
            seg_mask = (mask == seg_id)
            if seg_mask.sum() > 0:
                positions = np.argwhere(seg_mask)
                y_min, x_min = positions.min(axis=0)
                y_max, x_max = positions.max(axis=0)
                
                segments_info.append({
                    'id': seg_id,
                    'area': int(seg_mask.sum()),
                    'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                })
        
        print(f"Fallback found {len(segments_info)} segments")
        return mask, segments_info
    
    def visualize_segments(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Visualize segmentation with colors"""
        num_segments = mask.max() + 1
        
        # Use consistent random seed for reproducible colors
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_segments, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        
        colored_mask = colors[mask]
        result = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
        
        return result


# ============================================
# INTEGRATED 3D SEMANTIC SYSTEM
# ============================================
sam_checkpoint = "sam_vit_h_4b8939.pth"

class FoundationModels3D:
    """
    Modern 3D semantic understanding using foundation models
    """
    
    def __init__(self):
        self.depth_model = DepthAnythingV2(model_size='small')
        self.seg_model = RealSAM2(checkpoint_path=sam_checkpoint)
        print("\n" + "="*70)
        if self.seg_model.sam_available:
            print("REAL SAM 2 + Depth Anything V2 loaded!")
        else:
            print("Depth Anything V2 + Fallback Segmentation loaded")
        print("="*70)
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Process image with foundation models
        depth_map: Depth estimation
        seg_mask: Segmentation mask
        segments_info: Segment information
        """
        print("\n Processing image with foundation models...")
        
        # Depth estimation
        print(" [1/2] Estimating depth...")
        depth_map = self.depth_model.predict(image)
        print(f"Depth range: {depth_map.min():.3f} to {depth_map.max():.3f}")
        
        # Segmentation
        print("[2/2] Segmenting objects...")
        seg_mask, segments_info = self.seg_model.segment(image)
        
        return depth_map, seg_mask, segments_info
    
    def visualize(self, image, depth_map, seg_mask, segments_info):
        """Quick visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Depth
        depth_viz = axes[0, 1].imshow(depth_map, cmap='plasma')
        axes[0, 1].set_title('Depth Map', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(depth_viz, ax=axes[0, 1], fraction=0.046)
        
        # Segmentation
        seg_colored = self.seg_model.visualize_segments(image, seg_mask)
        axes[1, 0].imshow(cv2.cvtColor(seg_colored, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Segmentation ({len(segments_info)} segments)', 
                            fontweight='bold')
        axes[1, 0].axis('off')
        
        # Combined
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), 
                                         cv2.COLORMAP_PLASMA)
        combined = cv2.addWeighted(seg_colored, 0.5, depth_colored, 0.5, 0)
        axes[1, 1].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Semantic × Depth', fontweight='bold')
        axes[1, 1].axis('off')
        
        method = "SAM 2" if self.seg_model.sam_available else "Fallback"
        plt.suptitle(f'Foundation Models ({method})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # ============================================
# UTILITY FUNCTIONS
# ============================================

def check_and_install_sam2():
    """Check if SAM is available"""
    try:
        import segment_anything
        print("segment_anything package found")
        return True
    except ImportError:
        print("segment_anything not installed")
        print("\nTo install SAM:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
        return False

def check_depth_anything_v2():
    """Check if Depth Anything V2 is available"""
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        print("Depth Anything V2 package found")
        return True
    except ImportError:
        print("Depth Anything V2 not installed")
        print("\nTo install Depth Anything V2:")
        print("  git clone https://github.com/DepthAnything/Depth-Anything-V2")
        print("  cd Depth-Anything-V2")
        print("  pip install -r requirements.txt")
        return False



# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    IMAGE_PATH = "images/test_image.jpg"
    image = cv2.imread(IMAGE_PATH)
    
    # Resize
    scale = 0.5
    image = cv2.resize(image, None, fx=scale, fy=scale)
    print(f" Image loaded: {image.shape}")
    
    system = FoundationModels3D()
    
    # Process
    depth_map, seg_mask, segments_info = system.process(image)
    
    # Visualize
    system.visualize(image, depth_map, seg_mask, segments_info)
    
    print("\n" + "="*70)
    print(" DONE!")