"""
Model Evaluation Script for ResNet-152 Depth Estimation.

This script evaluates the trained model using standard depth estimation metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error) 
- REL (Absolute Relative Error)
- Delta Thresholds (δ < 1.25, δ < 1.25², δ < 1.25³)

Uses NYU Depth V2 dataset from Kaggle Hub (streams without full download).

Usage:
    python evaluate_model.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import io
import requests

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ResNetDepthModel


class DepthMetrics:
    """Calculate standard depth estimation metrics."""
    
    @staticmethod
    def compute_errors(pred: np.ndarray, target: np.ndarray) -> dict:
        """
        Compute depth estimation metrics.
        
        Args:
            pred: Predicted depth map
            target: Ground truth depth map
            
        Returns:
            Dictionary of metrics
        """
        # Ensure valid depth values (avoid division by zero)
        valid_mask = (target > 0.001) & (pred > 0.001)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if len(pred) == 0:
            return None
        
        # Threshold accuracies (δ < threshold)
        thresh = np.maximum(pred / target, target / pred)
        delta1 = np.mean(thresh < 1.25)      # δ < 1.25
        delta2 = np.mean(thresh < 1.25 ** 2)  # δ < 1.25²
        delta3 = np.mean(thresh < 1.25 ** 3)  # δ < 1.25³
        
        # Error metrics
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        mae = np.mean(np.abs(pred - target))
        abs_rel = np.mean(np.abs(pred - target) / target)
        sq_rel = np.mean(((pred - target) ** 2) / target)
        
        # Log RMSE (scale-invariant)
        log_rmse = np.sqrt(np.mean((np.log(pred) - np.log(target)) ** 2))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'log_rmse': log_rmse,
            'delta1': delta1,
            'delta2': delta2,
            'delta3': delta3
        }


def load_model(model_path: str, device: torch.device) -> ResNetDepthModel:
    """Load the trained model."""
    model = ResNetDepthModel()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def evaluate_with_kaggle_dataset(model_path: str, num_samples: int = 100):
    """
    Evaluate model using NYU Depth V2 dataset from Kaggle Hub.
    Streams data without downloading the full dataset.
    
    Args:
        model_path: Path to the model weights
        num_samples: Number of samples to evaluate (default: 100)
    """
    import kagglehub
    import pandas as pd
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"DEPTH ESTIMATION MODEL EVALUATION")
    print(f"Dataset: NYU Depth V2 (Kaggle Hub)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Evaluating on: {num_samples} samples")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found: {model_path}")
        return
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_path, device)
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Preprocessing transform
    transform = T.Compose([
        T.Resize((480, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    depth_transform = T.Compose([
        T.Resize((480, 640)),
        T.ToTensor(),
    ])
    
    print("\n" + "-"*60)
    print("Loading dataset from Kaggle Hub...")
    print("-"*60)
    
    # Set Kaggle credentials (use environment variables or kaggle.json)
    os.environ['KAGGLE_USERNAME'] = os.environ.get('KAGGLE_USERNAME', 'meetparmar40')
    os.environ['KAGGLE_KEY'] = os.environ.get('KAGGLE_KEY', '')
    
    try:
        # Download dataset (kagglehub caches it)
        dataset_path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2")
        print(f"Dataset path: {dataset_path}")
        
        # Find the CSV file
        csv_file = None
        for root, dirs, files in os.walk(dataset_path):
            if 'nyu2_train.csv' in files:
                csv_file = os.path.join(root, 'nyu2_train.csv')
                break
            elif 'nyu2_test.csv' in files:
                csv_file = os.path.join(root, 'nyu2_test.csv')
                break
        
        if csv_file is None:
            # Try to find any csv
            for root, dirs, files in os.walk(dataset_path):
                for f in files:
                    if f.endswith('.csv'):
                        csv_file = os.path.join(root, f)
                        break
                if csv_file:
                    break
        
        if csv_file is None:
            print("❌ Could not find CSV file in dataset")
            return
            
        print(f"Found CSV: {csv_file}")
        
        # Load CSV
        df = pd.read_csv(csv_file, header=None)
        print(f"Total samples in dataset: {len(df)}")
        
        # Debug: Print sample path from CSV
        sample_path_csv = df.iloc[0, 0]
        print(f"Sample CSV entry: {sample_path_csv}")
        
        # Find image root (auto-calibrate paths)
        # The CSV paths look like: /nyu_data/data/nyu2_train/basement_0001a_out/1.jpg
        # We need to find where these files actually are
        
        sample_filename = os.path.basename(sample_path_csv)
        sample_parent = os.path.basename(os.path.dirname(sample_path_csv))
        
        print(f"Looking for: {sample_filename} in folder {sample_parent}")
        
        img_root = None
        found_real_path = None
        
        for root, dirs, files in os.walk(dataset_path):
            if sample_filename in files:
                # Check if parent folder matches
                if os.path.basename(root) == sample_parent:
                    found_real_path = os.path.join(root, sample_filename)
                    break
        
        if found_real_path:
            print(f"Found actual file at: {found_real_path}")
            # Calculate the root by removing the CSV path suffix
            csv_rel_path = sample_path_csv.lstrip('/\\').replace('/', os.sep)
            if found_real_path.endswith(sample_filename):
                # Find where the CSV relative path starts in the real path
                real_path_normalized = found_real_path.replace('/', os.sep)
                csv_path_normalized = csv_rel_path.replace('/', os.sep)
                
                # Try to find the common suffix
                idx = real_path_normalized.find(csv_path_normalized)
                if idx != -1:
                    img_root = real_path_normalized[:idx]
                else:
                    # Alternative: just use dataset_path and strip nyu_data/data prefix
                    img_root = dataset_path
        
        if img_root is None:
            img_root = dataset_path
            
        print(f"Image root: {img_root}")
        
        # Test if path construction works
        test_row = df.iloc[0]
        test_img_rel = test_row[0].lstrip('/\\').replace('/', os.sep)
        test_img_path = os.path.join(img_root, test_img_rel)
        print(f"Test path: {test_img_path}")
        print(f"Test path exists: {os.path.exists(test_img_path)}")
        
        # If still not working, try different path construction
        if not os.path.exists(test_img_path):
            # Try joining with nyu_data folder
            nyu_data_path = os.path.join(dataset_path, 'nyu_data')
            if os.path.exists(nyu_data_path):
                # CSV paths start with /nyu_data/..., so strip that
                test_img_rel2 = test_row[0].replace('/nyu_data/', '').replace('/', os.sep)
                test_img_path2 = os.path.join(nyu_data_path, test_img_rel2)
                print(f"Alternative test path: {test_img_path2}")
                if os.path.exists(test_img_path2):
                    img_root = nyu_data_path
                    print(f"Using nyu_data as root: {img_root}")
        
        # Sample random indices for evaluation
        np.random.seed(42)
        total_samples = len(df)
        sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        
        # Metrics accumulator
        all_metrics = []
        valid_count = 0
        
        print(f"\nEvaluating {len(sample_indices)} samples...")
        print("-"*60)
        
        for i, idx in enumerate(tqdm(sample_indices, desc="Processing")):
            try:
                row = df.iloc[idx]
                # Handle path construction - strip leading slash and convert to OS path
                img_csv_path = row[0]
                depth_csv_path = row[1]
                
                # Try different path constructions
                # Option 1: Strip /nyu_data/ prefix if img_root is nyu_data folder
                if 'nyu_data' in str(img_root):
                    img_rel = img_csv_path.replace('/nyu_data/', '').replace('/', os.sep)
                    depth_rel = depth_csv_path.replace('/nyu_data/', '').replace('/', os.sep)
                else:
                    img_rel = img_csv_path.lstrip('/\\').replace('/', os.sep)
                    depth_rel = depth_csv_path.lstrip('/\\').replace('/', os.sep)
                
                img_path = os.path.join(img_root, img_rel)
                depth_path = os.path.join(img_root, depth_rel)
                
                # Debug first few paths
                if i < 3:
                    print(f"  Sample {i}: img={img_path}, exists={os.path.exists(img_path)}")
                
                if not os.path.exists(img_path) or not os.path.exists(depth_path):
                    continue
                
                # Load images
                image = Image.open(img_path).convert('RGB')
                depth_gt = Image.open(depth_path)
                
                # Preprocess
                input_tensor = transform(image).unsqueeze(0).to(device)
                depth_gt_tensor = depth_transform(depth_gt)
                
                # Scale ground truth depth (NYU depth is in mm, convert to meters)
                # The dataset stores depth as uint16, typical range 0-10000 (mm)
                depth_gt_np = np.array(depth_gt).astype(np.float32)
                
                # Normalize depth to meters (assuming max depth ~10m)
                if depth_gt_np.max() > 100:  # Likely in mm
                    depth_gt_np = depth_gt_np / 1000.0  # Convert mm to m
                elif depth_gt_np.max() > 10:  # Might be in cm
                    depth_gt_np = depth_gt_np / 100.0
                
                # Resize ground truth to match model output
                depth_gt_resized = np.array(Image.fromarray(depth_gt_np).resize((640, 480), Image.BILINEAR))
                
                # Predict
                with torch.no_grad():
                    pred = model(input_tensor)
                    # Resize prediction to match ground truth
                    pred = F.interpolate(pred, size=(480, 640), mode='bilinear', align_corners=True)
                
                pred_np = pred.squeeze().cpu().numpy()
                
                # Compute metrics
                metrics = DepthMetrics.compute_errors(pred_np, depth_gt_resized)
                
                if metrics is not None:
                    all_metrics.append(metrics)
                    valid_count += 1
                    
            except Exception as e:
                continue
        
        if len(all_metrics) == 0:
            print("\n❌ No valid samples could be evaluated")
            return
        
        # Aggregate metrics
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS ({valid_count} valid samples)")
        print(f"{'='*60}")
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        print("\n┌─────────────────┬──────────────┬──────────────┐")
        print("│ Metric          │    Value     │   Std Dev    │")
        print("├─────────────────┼──────────────┼──────────────┤")
        print(f"│ RMSE (m)        │ {avg_metrics['rmse']:>10.4f}   │ {avg_metrics['rmse_std']:>10.4f}   │")
        print(f"│ MAE (m)         │ {avg_metrics['mae']:>10.4f}   │ {avg_metrics['mae_std']:>10.4f}   │")
        print(f"│ Abs Rel         │ {avg_metrics['abs_rel']:>10.4f}   │ {avg_metrics['abs_rel_std']:>10.4f}   │")
        print(f"│ Sq Rel          │ {avg_metrics['sq_rel']:>10.4f}   │ {avg_metrics['sq_rel_std']:>10.4f}   │")
        print(f"│ Log RMSE        │ {avg_metrics['log_rmse']:>10.4f}   │ {avg_metrics['log_rmse_std']:>10.4f}   │")
        print("├─────────────────┼──────────────┼──────────────┤")
        print(f"│ δ < 1.25        │ {avg_metrics['delta1']*100:>9.2f}%   │ {avg_metrics['delta1_std']*100:>9.2f}%   │")
        print(f"│ δ < 1.25²       │ {avg_metrics['delta2']*100:>9.2f}%   │ {avg_metrics['delta2_std']*100:>9.2f}%   │")
        print(f"│ δ < 1.25³       │ {avg_metrics['delta3']*100:>9.2f}%   │ {avg_metrics['delta3_std']*100:>9.2f}%   │")
        print("└─────────────────┴──────────────┴──────────────┘")
        
        print("\n" + "-"*60)
        print("INTERPRETATION:")
        print("-"*60)
        print("""
Lower is better: RMSE, MAE, Abs Rel, Sq Rel, Log RMSE
Higher is better: δ thresholds (percentage of pixels within threshold)

Benchmark Comparison (NYU Depth V2):
┌──────────────────┬─────────┬─────────┬─────────┐
│ Model            │ Abs Rel │ RMSE    │ δ<1.25  │
├──────────────────┼─────────┼─────────┼─────────┤
│ Eigen et al.     │ 0.158   │ 0.641   │ 76.9%   │
│ Laina et al.     │ 0.127   │ 0.573   │ 81.1%   │
│ BTS (SOTA)       │ 0.110   │ 0.392   │ 88.5%   │
│ Your Model       │ {:.3f}   │ {:.3f}   │ {:.1f}%   │
└──────────────────┴─────────┴─────────┴─────────┘
""".format(avg_metrics['abs_rel'], avg_metrics['rmse'], avg_metrics['delta1']*100))
        
        print("✅ Evaluation complete!")
        
        return avg_metrics
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Find model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Check both locations
    model_paths = [
        os.path.join(script_dir, "resnet152_depth_model.pth"),
        os.path.join(parent_dir, "resnet152_depth_model.pth"),
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ Could not find resnet152_depth_model.pth")
        print(f"Searched: {model_paths}")
    else:
        # Evaluate using Kaggle dataset
        # Change num_samples for more thorough evaluation (default: 100)
        evaluate_with_kaggle_dataset(model_path, num_samples=100)
