# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# import torchvision.transforms as transforms

# class GradCAM:
#     def __init__(self, model, target_layer):
#         """
#         Initialize GradCAM for EdgeFace model
        
#         Args:
#             model: Your trained EdgeFace model
#             target_layer: The layer to visualize (typically last conv layer)
#         """
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_full_backward_hook(self.save_gradient)
    
#     def save_activation(self, module, input, output):
#         """Hook to save forward pass activations"""
#         self.activations = output.detach()
    
#     def save_gradient(self, module, grad_input, grad_output):
#         """Hook to save backward pass gradients"""
#         self.gradients = grad_output[0].detach()
    
#     def generate_cam(self, input_image, embedding_index=None):
#         """
#         Generate GradCAM heatmap for face recognition model
        
#         Args:
#             input_image: Input tensor [1, C, H, W]
#             embedding_index: Which embedding dimension to visualize (None for mean)
        
#         Returns:
#             cam: GradCAM heatmap as numpy array
#         """
#         # Forward pass
#         self.model.eval()
#         output = self.model(input_image)
        
#         # For face recognition, we visualize embeddings
#         # If no specific dimension, use mean of all embeddings
#         self.model.zero_grad()
        
#         if embedding_index is None:
#             # Visualize overall face representation
#             target = output.mean()
#         else:
#             # Visualize specific embedding dimension
#             target = output[0, embedding_index]
        
#         target.backward()
        
#         # Get gradients and activations
#         gradients = self.gradients[0]  # [C, H, W]
#         activations = self.activations[0]  # [C, H, W]
        
#         # Global average pooling on gradients
#         weights = gradients.mean(dim=(1, 2))  # [C]
        
#         # Weighted combination of activation maps
#         cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
#         for i, w in enumerate(weights):
#             cam += w * activations[i]
        
#         # Apply ReLU
#         cam = F.relu(cam)
        
#         # Normalize
#         cam = cam.cpu().numpy()
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
#         return cam
    
#     def visualize(self, input_image, original_image, embedding_index=None,
#                   alpha=0.4, colormap=cv2.COLORMAP_JET):
#         """
#         Create GradCAM overlay visualization
        
#         Args:
#             input_image: Preprocessed input tensor [1, C, H, W]
#             original_image: Original PIL Image or numpy array
#             embedding_index: Embedding dimension to visualize
#             alpha: Overlay transparency (0-1)
#             colormap: OpenCV colormap for heatmap
        
#         Returns:
#             overlayed_image: GradCAM overlay as numpy array
#             cam: Raw heatmap
#         """
#         # Generate CAM
#         cam = self.generate_cam(input_image, embedding_index)
        
#         # Convert original image to numpy if needed
#         if isinstance(original_image, Image.Image):
#             original_image = np.array(original_image)
        
#         # Resize CAM to match original image size
#         h, w = original_image.shape[:2]
#         cam_resized = cv2.resize(cam, (w, h))
        
#         # Apply colormap
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
#         # Ensure original image is RGB
#         if len(original_image.shape) == 2:
#             original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
#         # Overlay heatmap on original image
#         overlayed = heatmap * alpha + original_image * (1 - alpha)
#         overlayed = np.uint8(overlayed)
        
#         return overlayed, cam_resized


# def get_edgeface_transform():
#     """
#     Standard preprocessing for EdgeFace models
#     Adjust based on your training configuration
#     """
#     transform = transforms.Compose([
#         transforms.Resize((112, 112)),  # EdgeFace typically uses 112x112
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     return transform


# def plot_edgeface_gradcam(original_img, overlayed_img, cam, 
#                           embedding_dim=None, title_suffix=""):
#     """
#     Plot GradCAM results for EdgeFace
    
#     Args:
#         original_img: Original image
#         overlayed_img: GradCAM overlay
#         cam: Raw heatmap
#         embedding_dim: Embedding dimension visualized
#         title_suffix: Additional title text
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     # Original image
#     axes[0].imshow(original_img)
#     axes[0].set_title('Original Face Image')
#     axes[0].axis('off')
    
#     # GradCAM overlay
#     axes[1].imshow(overlayed_img)
#     title = f'EdgeFace GradCAM{title_suffix}'
#     if embedding_dim is not None:
#         title += f'\nEmbedding Dim: {embedding_dim}'
#     axes[1].set_title(title)
#     axes[1].axis('off')
    
#     # Heatmap only
#     axes[2].imshow(cam, cmap='jet')
#     axes[2].set_title('Face Attention Map')
#     axes[2].axis('off')
    
#     plt.tight_layout()
#     return fig


# def find_last_conv_layer(model):
#     """
#     Automatically find the last convolutional layer in EdgeFace
#     """
#     return model.model.stages[3].blocks[2].convs[2]

# def get_multi_layer_targets(model):
#     """
#     Manually define key convolutional layers across all stages for visualization.
#     Returns a dictionary of {name: layer_module}
#     """
#     # NOTE: Accessing via .model. is corrected here, assuming it's needed 
#     # based on the previous error context (TimmFRWrapperV2).
#     # If the stages are directly on the model, remove '.model'.
#     base = model.model if hasattr(model, 'model') else model

#     targets = {
#         'Stage 0': base.stages[0].blocks[2].conv_dw,
#         'Stage 1': base.stages[1].blocks[2].convs[0], 
#         'Stage 2': base.stages[2].blocks[8].convs[1],
#         'Stage 3': base.stages[3].blocks[2].convs[2],
#     }
#     return targets

# def plot_multi_layer_gradcam(original_img, results_dict, title_suffix=""):
#     """
#     Plots GradCAM results from multiple layers on a single figure.
#     """
#     num_layers = len(results_dict)
#     fig, axes = plt.subplots(1, num_layers + 1, figsize=(3 * (num_layers + 1), 4))
    
#     # 1. Original Image
#     axes[0].imshow(original_img)
#     axes[0].set_title('Original Image')
#     axes[0].axis('off')

#     # 2. Layer Overlays
#     i = 1
#     for layer_name, (overlayed_img, cam) in results_dict.items():
#         axes[i].imshow(overlayed_img)
#         axes[i].set_title(f'Layer: {layer_name}')
#         axes[i].axis('off')
#         i += 1
    
#     fig.suptitle(f'Multi-Layer EdgeFace GradCAM {title_suffix}', fontsize=16)
#     plt.tight_layout()
#     return fig

# # def print_final_stage_layers(model):
# #     print("--- Inspecting Final Stage (Stage 3) ---")
# #     for name, module in model.named_modules():
# #         # Look for anything inside the final stage
# #         if name.startswith('model.model.stages.3') and not isinstance(module, (nn.Sequential, nn.ModuleList)):
# #             print(f"Layer: {name} | Type: {module.__class__.__name__}")

# # # Run this function after successful model loading in your GradCAM script
# # # print_final_stage_layers(model)           


# # Main usage function for EdgeFace
# def visualize_edgeface_faces(image_paths, weights_path, save_dir='gradcam_results'):
#     """
#     Complete pipeline to visualize EdgeFace model attention on face images
    
#     Args:
#         image_paths: List of paths to face images
#         weights_path: Path to your trained model weights (.pth or .pt file)
#         save_dir: Directory to save results
#     """
#     import os
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Load EdgeFace model architecture
#     print("Loading EdgeFace model architecture...")
#     model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', 
#                           source='github', pretrained=False)
    
#     # Load your trained weights
#     print(f"Loading trained weights from {weights_path}...")
#     checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
#     # Handle different checkpoint formats
#     if isinstance(checkpoint, dict):
#         if 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         elif 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         else:
#             state_dict = checkpoint
#     else:
#         state_dict = checkpoint
    
#     # Fix key names if they have extra 'model.' prefix
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         # Remove extra 'model.' prefix if it exists
#         if key.startswith('model.model.'):
#             new_key = key.replace('model.model.', 'model.')
#         else:
#             new_key = key
#         new_state_dict[new_key] = value
    
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     print("Model loaded successfully!")

#     # Find the target layers for multi-layer visualization
#     target_layer = get_multi_layer_targets(model)
#     print(f"Using {len(target_layer)} layers for GradCAM.")
    
#     # Initialize GradCAM
#     gradcam = GradCAM(model, target_layer)
#     # Get preprocessing transform
#     transform = get_edgeface_transform()
    
#     # Process each image
#     for i, img_path in enumerate(image_paths):
#         print(f"Processing {img_path} with multi-layer CAM...")
        
#         # Load and preprocess image
#         original_img = Image.open(img_path).convert('RGB')
#         input_tensor = transform(original_img).unsqueeze(0)
        
#         results = {}
        
#         # Loop through each target layer
#         for layer_name, layer_module in target_layer_modules.items():
#             # Initialize GradCAM for the specific layer
#             gradcam = GradCAM(model, layer_module)
            
#             # Generate GradCAM
#             overlayed, cam = gradcam.visualize(
#                 input_tensor,
#                 original_img,
#                 alpha=0.4
#             )
#             results[layer_name] = (overlayed, cam)
        
#         # Plot and save
#         fig = plot_multi_layer_gradcam(
#             np.array(original_img),
#             results,
#             title_suffix=f" - Image {i+1}"
#         )
        
#         output_path = os.path.join(save_dir, f'edgeface_gradcam_multi_layer_{i}.png')
#         plt.savefig(output_path, bbox_inches='tight', dpi=150)
#         plt.close()
        
#         print(f"Saved: {output_path}")


# # Example: Single image visualization
# def visualize_single_face(image_path, weights_path, model=None):
#     """
#     Quick visualization of a single face image with your trained weights
    
#     Args:
#         image_path: Path to face image
#         weights_path: Path to your trained model weights
#         model: Pre-loaded EdgeFace model (optional)
#     """
#     # Load model if not provided
#     if model is None:
#         print("Loading EdgeFace model architecture...")
#         model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05',
#                               source='github', pretrained=False)
        
#         print(f"Loading trained weights from {weights_path}...")
#         checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
#         # Handle different checkpoint formats
#         if isinstance(checkpoint, dict):
#             if 'model_state_dict' in checkpoint:
#                 state_dict = checkpoint['model_state_dict']
#             elif 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             else:
#                 state_dict = checkpoint
#         else:
#             state_dict = checkpoint
        
#         # Fix key names if they have extra 'model.' prefix
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             # Remove extra 'model.' prefix if it exists
#             if key.startswith('model.model.'):
#                 new_key = key.replace('model.model.', 'model.')
#             else:
#                 new_key = key
#             new_state_dict[new_key] = value
        
#         model.load_state_dict(new_state_dict)
#         model.eval()
#         print("Model loaded successfully!")
    
#     # Setup GradCAM
#     target_layer = find_last_conv_layer(model)
#     gradcam = GradCAM(model, target_layer)
    
#     # Load and preprocess
#     original_img = Image.open(image_path).convert('RGB')
#     transform = get_edgeface_transform()
#     input_tensor = transform(original_img).unsqueeze(0)
    
#     # Generate visualization
#     overlayed, cam = gradcam.visualize(input_tensor, original_img)
    
#     # Display
#     fig = plot_edgeface_gradcam(np.array(original_img), overlayed, cam)
#     plt.show()
    
#     return overlayed, cam


# # Example usage
# if __name__ == "__main__":
#     # IMPORTANT: Replace with your actual weights path
#     YOUR_WEIGHTS_PATH = '/home/aryan/Desktop/Adl_assignment_1/model_A_best.pth'
    
#     # Example 1: Single image
#     image_list = ['masked_dataset/train/010/S2010L01.jpg', 
#                   'masked_dataset/test/012/S2012R01.jpg',
#                   'masked_dataset/train/015/S2015L02.jpg']
#     visualize_edgeface_faces(image_list, YOUR_WEIGHTS_PATH, save_dir='my_multi_layer_results')
    
#     # Example 2: Multiple images
#     # image_list = ['face1.jpg', 'face2.jpg', 'face3.jpg']
#     # visualize_edgeface_faces(image_list, YOUR_WEIGHTS_PATH, save_dir='my_gradcam_results')
    
#     # print("EdgeFace GradCAM ready!")
#     # print("\nUsage:")
#     # print("1. Single image: visualize_single_face('path/to/face.jpg', 'path/to/weights.pth')")
#     # print("2. Multiple images: visualize_edgeface_faces(['face1.jpg'], 'path/to/weights.pth')")

#============================================================================================

# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# import torchvision.transforms as transforms
# import os
# import torch.nn as nn # Added for module inspection/type check

# class GradCAM:
#     def __init__(self, model, target_layer):
#         """
#         Initialize GradCAM for EdgeFace model
        
#         Args:
#             model: Your trained EdgeFace model
#             target_layer: The layer to visualize (typically last conv layer)
#         """
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_full_backward_hook(self.save_gradient)
    
#     def save_activation(self, module, input, output):
#         """Hook to save forward pass activations"""
#         self.activations = output.detach()
    
#     def save_gradient(self, module, grad_input, grad_output):
#         """Hook to save backward pass gradients"""
#         self.gradients = grad_output[0].detach()
    
#     def generate_cam(self, input_image, embedding_index=None):
#         """
#         Generate GradCAM heatmap for face recognition model
        
#         Args:
#             input_image: Input tensor [1, C, H, W]
#             embedding_index: Which embedding dimension to visualize (None for mean)
        
#         Returns:
#             cam: GradCAM heatmap as numpy array
#         """
#         # Forward pass
#         self.model.eval()
#         output = self.model(input_image)
        
#         # For face recognition, we visualize embeddings
#         # If no specific dimension, use mean of all embeddings
#         self.model.zero_grad()
        
#         if embedding_index is None:
#             # Visualize overall face representation
#             target = output.mean()
#         else:
#             # Visualize specific embedding dimension
#             target = output[0, embedding_index]
        
#         target.backward()
        
#         # Get gradients and activations
#         gradients = self.gradients[0]  # [C, H, W]
#         activations = self.activations[0]  # [C, H, W]
        
#         # Global average pooling on gradients
#         weights = gradients.mean(dim=(1, 2))  # [C]
        
#         # Weighted combination of activation maps
#         cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
#         for i, w in enumerate(weights):
#             cam += w * activations[i]
        
#         # Apply ReLU
#         cam = F.relu(cam)
        
#         # Normalize
#         cam = cam.cpu().numpy()
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
#         return cam
    
#     def visualize(self, input_image, original_image, embedding_index=None,
#                   alpha=0.4, colormap=cv2.COLORMAP_JET):
#         """
#         Create GradCAM overlay visualization
#         """
#         # Generate CAM
#         cam = self.generate_cam(input_image, embedding_index)
        
#         # Convert original image to numpy if needed
#         if isinstance(original_image, Image.Image):
#             original_image = np.array(original_image)
        
#         # Resize CAM to match original image size
#         h, w = original_image.shape[:2]
#         cam_resized = cv2.resize(cam, (w, h))
        
#         # Apply colormap
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
#         # Ensure original image is RGB
#         if len(original_image.shape) == 2:
#             original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
#         # Overlay heatmap on original image
#         overlayed = heatmap * alpha + original_image * (1 - alpha)
#         overlayed = np.uint8(overlayed)
        
#         return overlayed, cam_resized


# def get_edgeface_transform():
#     """
#     Standard preprocessing for EdgeFace models
#     """
#     transform = transforms.Compose([
#         transforms.Resize((112, 112)),  # EdgeFace typically uses 112x112
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     return transform


# def plot_edgeface_gradcam(original_img, overlayed_img, cam, 
#                           embedding_dim=None, title_suffix=""):
#     """
#     Plot GradCAM results for EdgeFace (Single Layer Plot)
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     # Original image
#     axes[0].imshow(original_img)
#     axes[0].set_title('Original Face Image')
#     axes[0].axis('off')
    
#     # GradCAM overlay
#     axes[1].imshow(overlayed_img)
#     title = f'EdgeFace GradCAM{title_suffix}'
#     if embedding_dim is not None:
#         title += f'\nEmbedding Dim: {embedding_dim}'
#     axes[1].set_title(title)
#     axes[1].axis('off')
    
#     # Heatmap only
#     axes[2].imshow(cam, cmap='jet')
#     axes[2].set_title('Face Attention Map')
#     axes[2].axis('off')
    
#     plt.tight_layout()
#     return fig


# def find_last_conv_layer(model):
#     """
#     Returns the target layer for single-layer visualization (Final Stage Conv)
#     """
#     # Using the path confirmed to work for the final stage's convolution
#     return model.model.stages[3].blocks[2].convs[2]

# # --- NEW/UPDATED MULTI-LAYER FUNCTIONS ---

# def get_multi_layer_targets(model):
#     """
#     Manually define key convolutional layers across all stages for visualization.
#     Returns a dictionary of {name: layer_module}
#     """
#     # Core model is nested inside 'model.model'
#     base = model.model
    
#     # These paths are based on the EdgeNeXt/EdgeFace architecture structure
#     targets = {
#         'Stage 0 (Early)': base.stages[0].blocks[2].conv_dw,
#         'Stage 1 (Mid)': base.stages[1].blocks[2].convs[0], 
#         'Stage 2 (Mid-High)': base.stages[2].blocks[8].convs[1],
#         'Stage 3 (Final)': base.stages[3].blocks[2].convs[2],
#     }
#     return targets

# def plot_multi_layer_gradcam(original_img, results_dict, title_suffix=""):
#     """
#     Plots GradCAM results from multiple layers on a single figure.
#     """
#     num_layers = len(results_dict)
#     # +1 column for the original image
#     fig, axes = plt.subplots(1, num_layers + 1, figsize=(3 * (num_layers + 1), 4))
    
#     # 1. Original Image
#     axes[0].imshow(original_img)
#     axes[0].set_title('Original Image')
#     axes[0].axis('off')

#     # 2. Layer Overlays
#     i = 1
#     for layer_name, (overlayed_img, cam) in results_dict.items():
#         axes[i].imshow(overlayed_img)
#         axes[i].set_title(f'{layer_name}')
#         axes[i].axis('off')
#         i += 1
    
#     fig.suptitle(f'Multi-Layer EdgeFace GradCAM {title_suffix}', fontsize=16)
#     plt.tight_layout()
#     return fig

# # --- END OF NEW/UPDATED MULTI-LAYER FUNCTIONS ---


# # Main usage function for EdgeFace
# def visualize_edgeface_faces(image_paths, weights_path, save_dir='gradcam_results'):
#     """
#     Complete pipeline to visualize EdgeFace model attention across multiple layers.
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Load EdgeFace model architecture
#     print("Loading EdgeFace model architecture...")
#     model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', 
#                           source='github', pretrained=False)
    
#     # Load your trained weights
#     print(f"Loading trained weights from {weights_path}...")
#     checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
#     # Handle different checkpoint formats
#     if isinstance(checkpoint, dict):
#         if 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         elif 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         else:
#             state_dict = checkpoint
#     else:
#         state_dict = checkpoint
    
#     # Fix key names if they have extra 'model.' prefix
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         # Remove extra 'model.' prefix if it exists
#         if key.startswith('model.model.'):
#             new_key = key.replace('model.model.', 'model.')
#         else:
#             new_key = key
#         new_state_dict[new_key] = value
    
#     # NOTE: Model Size Mismatch Check (Assumed to be fixed previously)
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     print("Model loaded successfully!")

#     # Find the target layer modules for multi-layer visualization
#     target_layer_modules = get_multi_layer_targets(model) # CORRECT VARIABLE NAME
#     print(f"Using {len(target_layer_modules)} layers for GradCAM.")
    
#     # Get preprocessing transform
#     transform = get_edgeface_transform()
    
#     # Process each image
#     for i, img_path in enumerate(image_paths):
#         print(f"Processing {img_path} with multi-layer CAM...")
        
#         # Load and preprocess image
#         original_img = Image.open(img_path).convert('RGB')
#         input_tensor = transform(original_img).unsqueeze(0)
        
#         results = {}
        
#         # Loop through each target layer
#         for layer_name, layer_module in target_layer_modules.items():
#             # 🔥 FIX: Initialize GradCAM INSIDE the loop with the single layer module
#             gradcam = GradCAM(model, layer_module)
            
#             # Generate GradCAM
#             overlayed, cam = gradcam.visualize(
#                 input_tensor,
#                 original_img,
#                 alpha=0.4
#             )
#             results[layer_name] = (overlayed, cam)
        
#         # Plot and save
#         fig = plot_multi_layer_gradcam(
#             np.array(original_img),
#             results,
#             title_suffix=f" - Image {i+1}"
#         )
        
#         output_path = os.path.join(save_dir, f'edgeface_gradcam_multi_layer_{i}.png')
#         plt.savefig(output_path, bbox_inches='tight', dpi=150)
#         plt.close()
        
#         print(f"Saved: {output_path}")


# # Example: Single image visualization (kept for original functionality, uses find_last_conv_layer)
# def visualize_single_face(image_path, weights_path, model=None):
#     """
#     Quick visualization of a single face image with your trained weights
#     """
#     # Load model if not provided (keeping this separate for modularity)
#     if model is None:
#         print("Loading EdgeFace model architecture...")
#         model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05',
#                               source='github', pretrained=False)
        
#         print(f"Loading trained weights from {weights_path}...")
#         checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
#         # ... (rest of loading logic)
#         if isinstance(checkpoint, dict):
#             if 'model_state_dict' in checkpoint:
#                 state_dict = checkpoint['model_state_dict']
#             elif 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             else:
#                 state_dict = checkpoint
#         else:
#             state_dict = checkpoint
        
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             if key.startswith('model.model.'):
#                 new_key = key.replace('model.model.', 'model.')
#             else:
#                 new_key = key
#             new_state_dict[new_key] = value
        
#         model.load_state_dict(new_state_dict)
#         model.eval()
#         print("Model loaded successfully!")
    
#     # Setup GradCAM
#     target_layer = find_last_conv_layer(model)
#     gradcam = GradCAM(model, target_layer)
    
#     # Load and preprocess
#     original_img = Image.open(image_path).convert('RGB')
#     transform = get_edgeface_transform()
#     input_tensor = transform(original_img).unsqueeze(0)
    
#     # Generate visualization
#     overlayed, cam = gradcam.visualize(input_tensor, original_img)
    
#     # Display
#     fig = plot_edgeface_gradcam(np.array(original_img), overlayed, cam)
#     plt.show()
    
#     return overlayed, cam


# # Example usage
# if __name__ == "__main__":
#     # IMPORTANT: Replace with your actual weights path
#     YOUR_WEIGHTS_PATH = '/home/aryan/Desktop/Adl_assignment_1/model_A_best.pth'
    
#     # Example 1: Multi-layer visualization for a list of images
#     image_list = [
#         'masked_dataset/train/010/S2010L01.jpg', 
#         'masked_dataset/test/012/S2012R01.jpg',
#         'masked_dataset/train/015/S2015L02.jpg'
#     ]
#     visualize_edgeface_faces(image_list, YOUR_WEIGHTS_PATH, save_dir='my_multi_layer_results')
    
#     print("EdgeFace Multi-Layer GradCAM processing complete.")


# ============================================================================================ 


# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# import torchvision.transforms as transforms
# import os
# import torch.nn as nn 

# class GradCAM:
#     def __init__(self, model, target_layer):
#         """
#         Initialize GradCAM for EdgeFace model
        
#         Args:
#             model: Your trained EdgeFace model
#             target_layer: The layer to visualize (typically last conv layer)
#         """
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_full_backward_hook(self.save_gradient)
    
#     def save_activation(self, module, input, output):
#         """Hook to save forward pass activations"""
#         self.activations = output.detach()
    
#     def save_gradient(self, module, grad_input, grad_output):
#         """Hook to save backward pass gradients"""
#         self.gradients = grad_output[0].detach()
    
#     def generate_cam(self, input_image, embedding_index=None):
#         """
#         Generate GradCAM heatmap for face recognition model
#         """
#         self.model.eval()
#         output = self.model(input_image)
        
#         self.model.zero_grad()
        
#         if embedding_index is None:
#             # Visualize overall face representation
#             target = output.mean()
#         else:
#             # Visualize specific embedding dimension
#             target = output[0, embedding_index]
        
#         target.backward()
        
#         # Get gradients and activations
#         gradients = self.gradients[0]  # Expected [C, H, W]
#         activations = self.activations[0]  # Expected [C, H, W]
        
#         # Global average pooling on gradients
#         # This will now only execute for tensors with C, H, W dimensions
#         weights = gradients.mean(dim=(1, 2))  # [C]
        
#         # Weighted combination of activation maps
#         cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
#         for i, w in enumerate(weights):
#             cam += w * activations[i]
        
#         # Apply ReLU
#         cam = F.relu(cam)
        
#         # Normalize
#         cam = cam.cpu().numpy()
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
#         return cam
    
#     def visualize(self, input_image, original_image, embedding_index=None,
#                   alpha=0.4, colormap=cv2.COLORMAP_JET):
#         """
#         Create GradCAM overlay visualization
#         """
#         cam = self.generate_cam(input_image, embedding_index)
        
#         if isinstance(original_image, Image.Image):
#             original_image = np.array(original_image)
        
#         h, w = original_image.shape[:2]
#         cam_resized = cv2.resize(cam, (w, h))
        
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
#         if len(original_image.shape) == 2:
#             original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
#         overlayed = heatmap * alpha + original_image * (1 - alpha)
#         overlayed = np.uint8(overlayed)
        
#         return overlayed, cam_resized


# def get_edgeface_transform():
#     """Standard preprocessing for EdgeFace models"""
#     transform = transforms.Compose([
#         transforms.Resize((112, 112)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     return transform


# def plot_edgeface_gradcam(original_img, overlayed_img, cam, 
#                           embedding_dim=None, title_suffix=""):
#     """
#     Plot GradCAM results for EdgeFace (Single Layer Plot)
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     axes[0].imshow(original_img)
#     axes[0].set_title('Original Face Image')
#     axes[0].axis('off')
    
#     axes[1].imshow(overlayed_img)
#     title = f'EdgeFace GradCAM{title_suffix}'
#     if embedding_dim is not None:
#         title += f'\nEmbedding Dim: {embedding_dim}'
#     axes[1].set_title(title)
#     axes[1].axis('off')
    
#     axes[2].imshow(cam, cmap='jet')
#     axes[2].set_title('Face Attention Map')
#     axes[2].axis('off')
    
#     plt.tight_layout()
#     return fig


# def find_last_conv_layer(model):
#     """
#     Returns the target layer for single-layer visualization (Final Stage Conv)
#     """
#     return model.model.stages[3].blocks[2].convs[2]


# def get_intra_stage_targets(model):
#     """
#     Defines key CONVOLUTIONAL layers within Stage 3.
#     Only includes Conv2d outputs to ensure a [C, H, W] shape for GradCAM.
#     """
#     base = model.model
    
#     # Selecting layers known to output a [C, H, W] feature map
#     # We rely only on Conv2d layers from blocks that are known to exist.
#     targets = {
#         'St3: Block 1 Conv': base.stages[3].blocks[1].conv_dw,
#         'St3: Block 2 Conv Pre-XCA': base.stages[3].blocks[2].convs[2], 
#     }

#     return targets

# def plot_multi_layer_gradcam(original_img, results_dict, title_suffix=""):
#     """
#     Plots GradCAM results from multiple layers on a single figure.
#     """
#     num_layers = len(results_dict)
#     fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))
    
#     # 1. Original Image
#     axes[0].imshow(original_img)
#     axes[0].set_title('Original Image')
#     axes[0].axis('off')

#     # 2. Layer Overlays
#     i = 1
#     for layer_name, (overlayed_img, cam) in results_dict.items():
#         axes[i].imshow(overlayed_img)
#         axes[i].set_title(f'{layer_name}')
#         axes[i].axis('off')
#         i += 1
    
#     fig.suptitle(f'Intra-Stage EdgeFace GradCAM {title_suffix}', fontsize=16)
#     plt.tight_layout()
#     return fig


# # Main usage function for EdgeFace
# def visualize_edgeface_faces(image_paths, weights_path, save_dir='gradcam_results'):
#     """
#     Complete pipeline to visualize EdgeFace model attention across multiple layers.
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Load EdgeFace model architecture
#     print("Loading EdgeFace model architecture...")
#     model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', 
#                           source='github', pretrained=False)
    
#     # Load your trained weights
#     print(f"Loading trained weights from {weights_path}...")
#     checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
#     # Handle different checkpoint formats
#     if isinstance(checkpoint, dict):
#         if 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         elif 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         else:
#             state_dict = checkpoint
#     else:
#         state_dict = checkpoint
    
#     # Fix key names if they have extra 'model.' prefix
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         if key.startswith('model.model.'):
#             new_key = key.replace('model.model.', 'model.')
#         else:
#             new_key = key
#         new_state_dict[new_key] = value
    
#     # Model loading and evaluation
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     print("Model loaded successfully!")

#     # Find the target layer modules for multi-layer visualization (using intra-stage)
#     target_layer_modules = get_intra_stage_targets(model) 
#     print(f"Using {len(target_layer_modules)} layers for fine-grained GradCAM analysis.")
    
#     # Get preprocessing transform
#     transform = get_edgeface_transform()
    
#     # Process each image
#     for i, img_path in enumerate(image_paths):
#         print(f"Processing {img_path} with intra-stage CAM...")
        
#         original_img = Image.open(img_path).convert('RGB')
#         input_tensor = transform(original_img).unsqueeze(0)
        
#         results = {}
        
#         # Loop through each target layer
#         for layer_name, layer_module in target_layer_modules.items():
#             # Correct Initialization: GradCAM is initialized inside the loop
#             gradcam = GradCAM(model, layer_module)
            
#             # Generate GradCAM
#             overlayed, cam = gradcam.visualize(
#                 input_tensor,
#                 original_img,
#                 alpha=0.4
#             )
#             results[layer_name] = (overlayed, cam)
        
#         # Plot and save
#         fig = plot_multi_layer_gradcam(
#             np.array(original_img),
#             results,
#             title_suffix=f" - Image {i+1}"
#         )
        
#         output_path = os.path.join(save_dir, f'edgeface_gradcam_intra_stage_{i}.png')
#         plt.savefig(output_path, bbox_inches='tight', dpi=150)
#         plt.close()
        
#         print(f"Saved: {output_path}")


# # Example: Single image visualization (kept for original functionality)
# def visualize_single_face(image_path, weights_path, model=None):
#     """
#     Quick visualization of a single face image with your trained weights
#     """
#     if model is None:
#         print("Loading EdgeFace model architecture...")
#         model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05',
#                               source='github', pretrained=False)
        
#         print(f"Loading trained weights from {weights_path}...")
#         checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
#         if isinstance(checkpoint, dict):
#             if 'model_state_dict' in checkpoint:
#                 state_dict = checkpoint['model_state_dict']
#             elif 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             else:
#                 state_dict = checkpoint
#         else:
#             state_dict = checkpoint
        
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             if key.startswith('model.model.'):
#                 new_key = key.replace('model.model.', 'model.')
#             else:
#                 new_key = key
#             new_state_dict[new_key] = value
        
#         model.load_state_dict(new_state_dict)
#         model.eval()
#         print("Model loaded successfully!")
    
#     target_layer = find_last_conv_layer(model)
#     gradcam = GradCAM(model, target_layer)
    
#     original_img = Image.open(image_path).convert('RGB')
#     transform = get_edgeface_transform()
#     input_tensor = transform(original_img).unsqueeze(0)
    
#     overlayed, cam = gradcam.visualize(input_tensor, original_img)
    
#     fig = plot_edgeface_gradcam(np.array(original_img), overlayed, cam)
#     plt.show()
    
#     return overlayed, cam


# # Example usage
# if __name__ == "__main__":
#     # IMPORTANT: Replace with your actual weights path
#     YOUR_WEIGHTS_PATH = '/home/aryan/Desktop/Adl_assignment_1/model_A_best.pth'
    
#     # Example 1: Multi-layer visualization for a list of images (using intra-stage)
#     image_list = [
#         'masked_dataset/train/010/S2010L01.jpg', 
#         'masked_dataset/test/012/S2012R01.jpg',
#         'masked_dataset/train/015/S2015L02.jpg'
#     ]
#     visualize_edgeface_faces(image_list, YOUR_WEIGHTS_PATH, save_dir='my_intra_stage_results')
    
#     print("EdgeFace Intra-Stage GradCAM processing complete.")


# ======================================================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torchvision.transforms as transforms
import os
import torch.nn as nn 

# --- GradCAM Class (No Change) ---
class GradCAM:
    def __init__(self, model, target_layer):
        """Initialize GradCAM for EdgeFace model"""
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        # CRITICAL CHECK: Ensure output is 3D (C, H, W) for subsequent ops
        if output.dim() == 4: # Standard output is [1, C, H, W]
             self.activations = output.squeeze(0).detach()
        elif output.dim() == 3: # Some modules output [C, H, W] directly
             self.activations = output.detach()
        else: # Handle non-spatial output that caused the error (e.g., [1, C])
             raise ValueError(f"Module output has incorrect spatial dimensions ({output.shape}) for GradCAM.")
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, embedding_index=None):
        """Generate GradCAM heatmap for face recognition model"""
        self.model.eval()
        output = self.model(input_image)
        
        self.model.zero_grad()
        
        if embedding_index is None:
            target = output.mean()
        else:
            target = output[0, embedding_index]
        
        target.backward()
        
        # Get gradients and activations
        # We rely on hooks to set self.gradients/activations correctly as [C, H, W]
        gradients = self.gradients[0] if self.gradients.dim() == 4 else self.gradients
        activations = self.activations
        
        # Global average pooling on gradients (Requires [C, H, W])
        if gradients.dim() < 3:
            # If a Conv2d was accidentally missed, raise a specific error
            raise ValueError("Target layer output is not spatial ([C, H, W]). Cannot calculate GradCAM weights.")

        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and Normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(self, input_image, original_image, embedding_index=None,
                  alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Create GradCAM overlay visualization"""
        # Catch internal dimension errors from non-spatial layers
        try:
            cam = self.generate_cam(input_image, embedding_index)
        except ValueError as e:
            print(f"Skipping visualization for target layer due to error: {e}")
            # Return a blank blue image for failed layers
            blank_img = np.array(original_image)
            if len(blank_img.shape) == 2: blank_img = cv2.cvtColor(blank_img, cv2.COLOR_GRAY2RGB)
            h, w = blank_img.shape[:2]
            return np.uint8(blank_img * 0.2), np.zeros((h, w)) # Dark image, black heatmap

        # ... (rest of visualization logic, unchanged)
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        overlayed = heatmap * alpha + original_image * (1 - alpha)
        overlayed = np.uint8(overlayed)
        
        return overlayed, cam_resized

# --- Utility Functions ---

def get_edgeface_transform():
    """Standard preprocessing for EdgeFace models"""
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

def plot_edgeface_gradcam(original_img, overlayed_img, cam, embedding_dim=None, title_suffix=""):
    """Plot GradCAM results for EdgeFace (Single Layer Plot)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img); axes[0].set_title('Original Face Image'); axes[0].axis('off')
    axes[1].imshow(overlayed_img)
    title = f'EdgeFace GradCAM{title_suffix}'
    if embedding_dim is not None: title += f'\nEmbedding Dim: {embedding_dim}'
    axes[1].set_title(title); axes[1].axis('off')
    axes[2].imshow(cam, cmap='jet'); axes[2].set_title('Face Attention Map'); axes[2].axis('off')
    plt.tight_layout()
    return fig

def find_last_conv_layer(model):
    """Returns the target layer for single-layer visualization (Final Stage Conv)"""
    return model.model.stages[3].blocks[2].convs[2]

# --- NEW/UPDATED Intra-Stage Target Selector ---

def is_spatial_module(module):
    """Heuristic check for modules that likely output a spatial feature map."""
    # Include all 2D Convolutions
    if isinstance(module, torch.nn.Conv2d):
        return True
    # Include other common feature extractors/transformers whose output might be reshaped back.
    # We will use this list to be permissive, but the GradCAM logic will filter them later.
    if module.__class__.__name__ in ['CrossCovarianceAttn', 'SplitTransposeBlock', 'Mlp']:
        return True
    # Exclude known non-spatial layers
    if isinstance(module, (torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.Identity)):
        return False
    return False

def get_intra_stage_targets(model, stage_index=3):
    """
    Dynamically define all spatial feature layers within a specific stage.
    """
    base = model.model
    targets = {}
    stage_prefix = f'model.stages.{stage_index}'
    
    # Iterate over all named modules
    for name, module in model.named_modules():
        if name.startswith(stage_prefix) and is_spatial_module(module):
            # Clean up the name for the plot title
            layer_name = name.replace(stage_prefix, f'St{stage_index}').replace('blocks.', 'B').replace('.', '_')
            targets[layer_name] = module
            
    # CRITICAL: If no spatial layers are found, add the entire stage as a fallback.
    if not targets:
        print(f"Warning: No specific spatial modules found in stage {stage_index}. Using entire stage.")
        targets[f'St{stage_index}_Full'] = base.stages[stage_index]
        
    return targets

def plot_multi_layer_gradcam(original_img, results_dict, title_suffix=""):
    """Plots GradCAM results from multiple layers on a single figure."""
    num_layers = len(results_dict)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(3 * (num_layers + 1), 4))
    
    axes[0].imshow(original_img); axes[0].set_title('Original Image'); axes[0].axis('off')

    i = 1
    for layer_name, (overlayed_img, cam) in results_dict.items():
        axes[i].imshow(overlayed_img); axes[i].set_title(f'{layer_name}'); axes[i].axis('off')
        i += 1
    
    fig.suptitle(f'Intra-Stage EdgeFace GradCAM {title_suffix}', fontsize=16)
    plt.tight_layout()
    return fig

# --- Main Functions ---

def visualize_edgeface_faces(image_paths, weights_path, save_dir='gradcam_results'):
    """
    Complete pipeline to visualize EdgeFace model attention across multiple layers.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading EdgeFace model architecture...")
    model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', source='github', pretrained=False)
    
    print(f"Loading trained weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # ... (Checkpoint loading and key fixing logic)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace('model.model.', 'model.')] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully!")

    # Use the dynamic layer finder for Stage 3
    target_layer_modules = get_intra_stage_targets(model, stage_index=3) 
    print(f"Using {len(target_layer_modules)} modules for granular GradCAM analysis in Stage 3.")
    
    transform = get_edgeface_transform()
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing {img_path} with intra-stage CAM...")
        
        original_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(original_img).unsqueeze(0)
        results = {}
        
        # Loop through each target layer
        for layer_name, layer_module in target_layer_modules.items():
            try:
                gradcam = GradCAM(model, layer_module)
                overlayed, cam = gradcam.visualize(input_tensor, original_img, alpha=0.4)
                results[layer_name] = (overlayed, cam)
            except ValueError as e:
                # Catch the dimension error from the GradCAM class
                print(f"Skipping {layer_name} due to dimension error: {e}")
                # Optional: Add a placeholder result to the plot
                # results[layer_name] = (np.zeros_like(np.array(original_img)), np.zeros((1, 1))) 
                
        # Plot and save
        if results:
            fig = plot_multi_layer_gradcam(
                np.array(original_img), results, title_suffix=f" - Image {i+1}"
            )
            output_path = os.path.join(save_dir, f'edgeface_gradcam_intra_stage_{i}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved: {output_path}")
        else:
            print(f"No successful visualizations generated for {img_path}.")


# Example: Single image visualization (kept for original functionality)
def visualize_single_face(image_path, weights_path, model=None):
    """Quick visualization of a single face image with your trained weights"""
    # ... (unchanged single visualization logic)
    if model is None:
        print("Loading EdgeFace model architecture...")
        model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05',
                              source='github', pretrained=False)
        
        print(f"Loading trained weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[key.replace('model.model.', 'model.')] = value
        
        model.load_state_dict(new_state_dict)
        model.eval()
        print("Model loaded successfully!")
    
    target_layer = find_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer)
    
    original_img = Image.open(image_path).convert('RGB')
    transform = get_edgeface_transform()
    input_tensor = transform(original_img).unsqueeze(0)
    
    overlayed, cam = gradcam.visualize(input_tensor, original_img)
    
    fig = plot_edgeface_gradcam(np.array(original_img), overlayed, cam)
    plt.show()
    
    return overlayed, cam


# Example usage
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual weights path
    YOUR_WEIGHTS_PATH = '/home/aryan/Desktop/Adl_assignment_1/model_A_best.pth'
    #YOUR_WEIGHTS_PATH = '/home/aryan/Desktop/Adl_assignment_1/edgeface/checkpoints/edgeface_s_gamma_05.pt'
    
    # Example 1: Multi-layer visualization for a list of images (using intra-stage)
    image_list = [
        'masked_dataset/train/010/S2010L01.jpg', 
        'masked_dataset/test/012/S2012R01.jpg',
        'masked_dataset/train/015/S2015L02.jpg'
    ]
    visualize_edgeface_faces(image_list, YOUR_WEIGHTS_PATH, save_dir='my_intra_stage_results')
    
    print("EdgeFace Intra-Stage GradCAM processing complete.")