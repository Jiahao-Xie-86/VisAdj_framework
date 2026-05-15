"""
SAM2 Encoder Wrapper

Provides frozen SAM2 encoder for feature extraction.
Uses SAM (v1) ImageEncoderViT as the backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import sys
import math
import inspect
from pathlib import Path


class _LoRA_qkv(nn.Module):
    """
    LoRA wrapper for QKV attention layer in SAM ViT.
    
    In SAM, the attention is implemented as:
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    LoRA adds low-rank adaptation: W' = W + B @ A
    where A is [dim, r] and B is [r, dim], with r << dim
    """
    
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        # Store original QKV weights (frozen)
        self.weight = qkv.weight
        self.bias = qkv.bias if qkv.bias is not None else None
        # LoRA adapters for Q and V (K is not adapted)
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        
    def forward(self, x):
        # Original QKV projection
        # x shape: [B, H, W, C] or [B, N, C] where C = dim
        # qkv shape: [B, H, W, 3*C] or [B, N, 3*C]
        qkv = F.linear(x, self.weight, self.bias)
        
        # Add LoRA adaptations for Q and V
        # new_q and new_v have same shape as x: [B, H, W, C] or [B, N, C]
        new_q = self.linear_b_q(self.linear_a_q(x))  # LoRA(Q)
        new_v = self.linear_b_v(self.linear_a_v(x))  # LoRA(V)
        
        # Add LoRA to Q (first C channels) and V (last C channels)
        # Handle both 3D [B, N, 3*C] and 4D [B, H, W, 3*C] cases
        if len(qkv.shape) == 4:
            # 4D case: [B, H, W, 3*C]
            qkv[:, :, :, :self.dim] += new_q
            qkv[:, :, :, -self.dim:] += new_v
        else:
            # 3D case: [B, N, 3*C]
            qkv[:, :, :self.dim] += new_q
            qkv[:, :, -self.dim:] += new_v
        
        return qkv

try:
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

# Add SAM path to sys.path if needed
# Try multiple import paths for SAM
try:
    # First try: SAM from baseline (if available)
    project_root = Path(__file__).parent.parent.parent
    sam_path = project_root / "baseline" / "sam_road" / "sam"
    if sam_path.exists():
        sys.path.insert(0, str(sam_path.parent))
        from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
    else:
        raise ImportError("SAM not in baseline")
except ImportError:
    try:
        # Second try: segment_anything package (from pip install)
        from segment_anything.modeling.image_encoder import ImageEncoderViT
    except ImportError:
        try:
            # Third try: sam.segment_anything (alternative path)
            from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
        except ImportError:
            raise ImportError(
                "Could not find SAM encoder. Please ensure SAM is installed.\n"
                "Options:\n"
                "1. Set PYTHONPATH: export PYTHONPATH=/path/to/baseline/sam_road/sam:$PYTHONPATH\n"
                "2. Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )


class SAM2Encoder(nn.Module):
    """
    Frozen SAM2 encoder wrapper for feature extraction.
    
    Extracts multi-scale features from images using SAM's ViT encoder.
    The encoder is frozen to prevent overfitting and reduce training complexity.
    """
    
    def __init__(
        self,
        sam_version: str = 'vit_b',
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        freeze: bool = True,
        image_size: int = 512
    ):
        """
        Args:
            sam_version: SAM version ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: Path to SAM checkpoint file
            config_path: Path to SAM config file (required for SAM2 variants)
            freeze: Whether to freeze encoder weights
            image_size: Input image size
        """
        super().__init__()
        
        self.sam_version = sam_version
        self.image_size = image_size
        self.freeze = freeze
        self.config_path = config_path
        self.using_sam2 = False
        self.using_sam3 = False
        self._feature_dim = 256  # default neck dim
        self.sam2_model = None
        self.sam3_model = None
        self.sam3_input_size = 1008
        self.sam3_feature_projections = nn.ModuleDict({
            "64": nn.Conv2d(64, self._feature_dim, kernel_size=1),
            "128": nn.Conv2d(128, self._feature_dim, kernel_size=1),
            "512": nn.Conv2d(512, self._feature_dim, kernel_size=1),
            "768": nn.Conv2d(768, self._feature_dim, kernel_size=1),
            "1024": nn.Conv2d(1024, self._feature_dim, kernel_size=1),
            "1280": nn.Conv2d(1280, self._feature_dim, kernel_size=1),
        })
        self.encoder_global_attn_indexes = None  # Will be set during encoder initialization
        self.use_lora = False
        self.lora_rank = None
        self.lora_w_As = []  # LoRA A matrices (down projection)
        self.lora_w_Bs = []  # LoRA B matrices (up projection)
        
        if sam_version.startswith('sam3'):
            self.using_sam3 = True
            self._init_sam3_encoder(checkpoint_path)
        elif config_path or sam_version.startswith('sam2'):
            self.using_sam2 = True
            self._init_sam2_encoder(config_path, checkpoint_path)
        else:
            self._init_sam_v1_encoder(sam_version, image_size, checkpoint_path)
        
        if freeze:
            self.freeze_encoder()
        
        # Pixel normalization (SAM standard)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
            persistent=False
        )
        
    def _init_sam_v1_encoder(self, sam_version: str, image_size: int, checkpoint_path: Optional[str]):
        # SAM configuration based on version
        if sam_version == 'vit_b':
            encoder_embed_dim = 768
            encoder_depth = 12
            encoder_num_heads = 12
            encoder_global_attn_indexes = [2, 5, 8, 11]
        elif sam_version == 'vit_l':
            encoder_embed_dim = 1024
            encoder_depth = 24
            encoder_num_heads = 16
            encoder_global_attn_indexes = [5, 11, 17, 23]
        elif sam_version == 'vit_h':
            encoder_embed_dim = 1280
            encoder_depth = 32
            encoder_num_heads = 16
            encoder_global_attn_indexes = [7, 15, 23, 31]
        else:
            raise ValueError(f"Unknown SAM version: {sam_version}")
        
        # Store global attention indexes for positional embedding resizing
        self.encoder_global_attn_indexes = encoder_global_attn_indexes
        
        self.encoder = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            num_heads=encoder_num_heads,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=self._feature_dim,
        )
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def _init_sam3_encoder(self, checkpoint_path: Optional[str]):
        """Initialize the official SAM3 image model and expose its vision backbone."""
        checkpoint_path = checkpoint_path or None
        try:
            from sam3.model_builder import build_sam3_image_model
        except ImportError:
            try:
                from sam3 import build_sam3_image_model
            except ImportError as exc:
                raise ImportError(
                    "SAM3 support requires the official facebookresearch/sam3 package.\n"
                    "Install it in the training environment, then rerun with --sam-version sam3.\n"
                    "Expected usage: pip install -e /path/to/facebookresearch/sam3"
                ) from exc

        build_kwargs = {
            "checkpoint_path": checkpoint_path,
            "device": "cpu",
            "eval_mode": True,
            "load_from_HF": checkpoint_path is None,
        }
        try:
            signature = inspect.signature(build_sam3_image_model)
            build_kwargs = {k: v for k, v in build_kwargs.items() if k in signature.parameters}
        except (TypeError, ValueError):
            build_kwargs = {k: v for k, v in build_kwargs.items() if v is not None}

        self.sam3_model = build_sam3_image_model(**build_kwargs)
        self.sam3_model.to(dtype=torch.bfloat16)
        self.sam3_model.eval()
        self.encoder = getattr(self.sam3_model, "backbone", self.sam3_model)
        if hasattr(self.encoder, "eval"):
            self.encoder.eval()

    def _init_sam2_encoder(self, config_path: Optional[str], checkpoint_path: Optional[str]):
        if not HYDRA_AVAILABLE:
            raise ImportError(
                "Hydra and OmegaConf are required for SAM2 support. Install with 'pip install hydra-core omegaconf'."
            )
        if config_path is None:
            raise ValueError(
                "SAM2 requires a config YAML. Provide --sam-config or pass config_path when constructing SAM2Encoder."
            )
        cfg = OmegaConf.load(config_path)
        # Instantiate full SAM2 model; we only use the image encoder
        # Use _recursive_=True to properly instantiate nested components like image_encoder
        self.sam2_model = instantiate(cfg.model, _recursive_=True)
        self.encoder = self.sam2_model.image_encoder
        self.encoder.eval()
        self.image_size = int(getattr(cfg.model, 'image_size', self.image_size))
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
 
    def resize_sam_pos_embed(
        self, 
        state_dict: Dict[str, torch.Tensor], 
        image_size: int, 
        vit_patch_size: int, 
        encoder_global_attn_indexes: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Resize positional embeddings when image size differs from checkpoint.
        
        Args:
            state_dict: State dict from checkpoint
            image_size: Target image size
            vit_patch_size: ViT patch size (typically 16)
            encoder_global_attn_indexes: List of layer indices with global attention
        
        Returns:
            Resized state dict
        """
        new_state_dict = {k: v for k, v in state_dict.items()}
        
        # Resize pos_embed if present
        pos_embed_key = None
        for k in ['pos_embed', 'image_encoder.pos_embed', 'encoder.pos_embed']:
            if k in new_state_dict:
                pos_embed_key = k
                break
        
        if pos_embed_key is not None:
            pos_embed = new_state_dict[pos_embed_key]
            token_size = int(image_size // vit_patch_size)
            
            # Check if resizing is needed
            # pos_embed shape: [1, H, W, C] or [1, H*W, C]
            if len(pos_embed.shape) == 4:
                # [1, H, W, C] format
                if pos_embed.shape[1] != token_size or pos_embed.shape[2] != token_size:
                    pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, C, H, W]
                    pos_embed = F.interpolate(
                        pos_embed, 
                        (token_size, token_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    pos_embed = pos_embed.permute(0, 2, 3, 1)  # [1, H, W, C]
                    new_state_dict[pos_embed_key] = pos_embed
            elif len(pos_embed.shape) == 3:
                # [1, H*W, C] format - reshape first
                hw = int(pos_embed.shape[1] ** 0.5)
                if hw != token_size:
                    pos_embed_4d = pos_embed.view(1, hw, hw, -1)  # [1, H, W, C]
                    pos_embed_4d = pos_embed_4d.permute(0, 3, 1, 2)  # [1, C, H, W]
                    pos_embed_4d = F.interpolate(
                        pos_embed_4d,
                        (token_size, token_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    pos_embed_4d = pos_embed_4d.permute(0, 2, 3, 1)  # [1, H, W, C]
                    new_state_dict[pos_embed_key] = pos_embed_4d.view(1, token_size * token_size, -1)
        
        # Resize relative position embeddings for all attention layers
        # Relative position embeddings have shape [2*token_size-1, 2*token_size-1]
        rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
        if rel_pos_keys:
            token_size = int(image_size // vit_patch_size)
            target_size = token_size * 2 - 1  # Target size for relative position embeddings
            
            for k in rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                if len(rel_pos_params.shape) == 2:
                    h, w = rel_pos_params.shape
                    # Resize both dimensions if they don't match
                    if h != target_size or w != target_size:
                        # Use bilinear interpolation to resize
                        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                        rel_pos_params = F.interpolate(
                            rel_pos_params,
                            (target_size, target_size),
                            mode='bilinear',
                            align_corners=False
                        )
                        new_state_dict[k] = rel_pos_params[0, 0, ...]
        
        return new_state_dict

    def load_checkpoint(self, checkpoint_path: str):
        """Load SAM checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if self.using_sam3:
            raise RuntimeError(
                "SAM3 checkpoints are loaded by build_sam3_image_model during initialization; "
                "direct load_checkpoint() is not used for SAM3."
            )
        elif self.using_sam2:
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            missing_keys, unexpected_keys = self.sam2_model.load_state_dict(
                state_dict, strict=False
            )
        else:
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Extract encoder state dict first
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('image_encoder.') or k.startswith('encoder.'):
                    new_key = k.replace('image_encoder.', '').replace('encoder.', '')
                    encoder_state_dict[new_key] = v
                elif not k.startswith('prompt_encoder.') and not k.startswith('mask_decoder.'):
                    # Direct encoder keys (no prefix) - but check if they're encoder-related
                    # Skip if they look like they belong to other components
                    if not any(x in k for x in ['prompt_encoder', 'mask_decoder', 'sam_predictor']):
                        encoder_state_dict[k] = v
            
            # Get model's expected state dict to determine target sizes
            model_state_dict = self.encoder.state_dict()
            
            # Filter and resize mismatched parameters, especially relative position embeddings
            filtered_state_dict = {}
            for k, v in encoder_state_dict.items():
                if k in model_state_dict:
                    if v.shape != model_state_dict[k].shape:
                        # Check if it's a relative position embedding
                        if 'rel_pos' in k:
                            # Resize relative position embeddings to match model's expected size
                            if len(v.shape) == 2 and len(model_state_dict[k].shape) == 2:
                                target_shape = model_state_dict[k].shape
                                # Use bilinear interpolation to resize
                                v_resized = v.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                                v_resized = F.interpolate(
                                    v_resized,
                                    (target_shape[0], target_shape[1]),
                                    mode='bilinear',
                                    align_corners=False
                                )
                                filtered_state_dict[k] = v_resized[0, 0, ...]
                                print(f"Resized relative position embedding {k} from {v.shape} to {target_shape}")
                            else:
                                # Skip if we can't resize
                                print(f"Warning: Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")
                        elif 'pos_embed' in k:
                            # Handle absolute positional embeddings
                            target_shape = model_state_dict[k].shape
                            if len(v.shape) == len(target_shape):
                                # Try to resize if dimensions match
                                if len(v.shape) == 4:
                                    # [1, H, W, C] format
                                    v_resized = v.permute(0, 3, 1, 2)  # [1, C, H, W]
                                    v_resized = F.interpolate(
                                        v_resized,
                                        (target_shape[1], target_shape[2]),
                                        mode='bilinear',
                                        align_corners=False
                                    )
                                    v_resized = v_resized.permute(0, 2, 3, 1)  # [1, H, W, C]
                                    filtered_state_dict[k] = v_resized
                                    print(f"Resized positional embedding {k} from {v.shape} to {target_shape}")
                                elif len(v.shape) == 3:
                                    # [1, H*W, C] format
                                    hw_src = int(v.shape[1] ** 0.5)
                                    hw_tgt = int(target_shape[1] ** 0.5)
                                    if hw_src != hw_tgt:
                                        v_4d = v.view(1, hw_src, hw_src, -1)  # [1, H, W, C]
                                        v_4d = v_4d.permute(0, 3, 1, 2)  # [1, C, H, W]
                                        v_4d = F.interpolate(
                                            v_4d,
                                            (hw_tgt, hw_tgt),
                                            mode='bilinear',
                                            align_corners=False
                                        )
                                        v_4d = v_4d.permute(0, 2, 3, 1)  # [1, H, W, C]
                                        filtered_state_dict[k] = v_4d.view(1, hw_tgt * hw_tgt, -1)
                                        print(f"Resized positional embedding {k} from {v.shape} to {target_shape}")
                                    else:
                                        filtered_state_dict[k] = v
                                else:
                                    print(f"Warning: Skipping {k} due to unexpected shape: {v.shape}")
                            else:
                                print(f"Warning: Skipping {k} due to shape mismatch: {v.shape} vs {target_shape}")
                        else:
                            # For other mismatches, skip (might be due to different model sizes)
                            print(f"Warning: Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")
                    else:
                        filtered_state_dict[k] = v
                else:
                    # Key not in model, skip it (might be from different SAM version)
                    pass
            
            missing_keys, unexpected_keys = self.encoder.load_state_dict(
                filtered_state_dict, strict=False
            )
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
 
    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
    
    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()
    
    def apply_lora(self, lora_rank: int = 8, lora_layer_selection: Optional[List[int]] = None):
        """
        Apply LoRA (Low-Rank Adaptation) to the encoder.
        
        Supports both SAM v1 (ViT) and SAM2 (Hiera) architectures:
        - SAM v1: LoRA applied to encoder.blocks[i].attn.qkv
        - SAM2: LoRA applied to encoder.trunk.blocks[i].attn.qkv
        
        LoRA adds trainable low-rank matrices to attention layers, allowing
        efficient fine-tuning with fewer parameters than full fine-tuning.
        
        Args:
            lora_rank: Rank of LoRA matrices (r in the paper). Lower rank = fewer parameters.
            lora_layer_selection: List of layer indices to apply LoRA to. 
                                 If None, applies to all transformer blocks.
        """
        self.use_lora = True
        self.lora_rank = lora_rank
        if self.using_sam3:
            raise NotImplementedError(
                "LoRA for SAM3 is not enabled yet because the official SAM3 backbone "
                "does not share a stable block.attn.qkv path with SAM/SAM2 in this wrapper. "
                "Run the SAM3 benchmark with the frozen encoder first."
            )
        
        # Freeze encoder first
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Get blocks based on architecture
        if self.using_sam2:
            # SAM2: blocks are in encoder.trunk.blocks (Hiera backbone)
            blocks = self.encoder.trunk.blocks
        else:
            # SAM v1: blocks are in encoder.blocks (ViT)
            blocks = self.encoder.blocks
        
        # Determine which layers to apply LoRA to
        if lora_layer_selection is None:
            lora_layer_selection = list(range(len(blocks)))
        
        self.lora_w_As = []
        self.lora_w_Bs = []
        
        # Apply LoRA to selected layers
        for layer_idx, block in enumerate(blocks):
            if layer_idx not in lora_layer_selection:
                continue
            
            # Get the QKV linear layer
            # Both SAM v1 (ViT) and SAM2 (Hiera) use block.attn.qkv
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features
            
            # Create LoRA adapters for Q and V (K is not adapted, following common practice)
            w_a_linear_q = nn.Linear(dim, lora_rank, bias=False)
            w_b_linear_q = nn.Linear(lora_rank, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, lora_rank, bias=False)
            w_b_linear_v = nn.Linear(lora_rank, dim, bias=False)
            
            # Store for parameter access
            self.lora_w_As.append(w_a_linear_q)
            self.lora_w_Bs.append(w_b_linear_q)
            self.lora_w_As.append(w_a_linear_v)
            self.lora_w_Bs.append(w_b_linear_v)
            
            # Replace QKV layer with LoRA wrapper
            # Register LoRA modules as submodules so they're properly tracked
            lora_wrapper = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
            block.attn.qkv = lora_wrapper
        
        # Initialize LoRA parameters
        # A matrices: Kaiming uniform (standard for down-projection)
        for w_A in self.lora_w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        
        # B matrices: zeros (so initial adaptation is zero, following LoRA paper)
        for w_B in self.lora_w_Bs:
            nn.init.zeros_(w_B.weight)
        
        # Move LoRA modules to same device as encoder
        if next(self.encoder.parameters()).is_cuda:
            for w_A, w_B in zip(self.lora_w_As, self.lora_w_Bs):
                w_A.cuda()
                w_B.cuda()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SAM encoder.
        
        Args:
            x: Input images [B, C, H, W] in range [0, 1]
        
        Returns:
            Image embeddings [B, D, H', W'] where H'=H/16, W'=W/16
        """
        # Normalize pixels
        x = (x - self.pixel_mean) / self.pixel_std
        encoder_input = x
        if self.using_sam3 and x.shape[-2:] != (self.sam3_input_size, self.sam3_input_size):
            encoder_input = F.interpolate(
                x,
                size=(self.sam3_input_size, self.sam3_input_size),
                mode="bilinear",
                align_corners=False,
            )
        
        # Encode
        if self.freeze:
            with torch.no_grad():
                features = self._forward_sam3_image(encoder_input.to(torch.bfloat16)) if self.using_sam3 else self.encoder(encoder_input)
        else:
            features = self._forward_sam3_image(encoder_input.to(torch.bfloat16)) if self.using_sam3 else self.encoder(encoder_input)
        
        if self.using_sam3:
            features = self._extract_sam3_spatial_feature(features)
            features = self._adapt_sam3_feature_map(features, x)
            return features

        if isinstance(features, dict):
            if 'backbone_fpn' in features:
                features = features['backbone_fpn'][0]
            elif 'image_features' in features:
                features = features['image_features']
            else:
                features = next(iter(features.values()))
        if isinstance(features, (list, tuple)):
            features = features[0]
        
        return features

    def _forward_sam3_image(self, x: torch.Tensor):
        """Call the SAM3 image backbone using the most common official entry points."""
        if self.sam3_model is not None:
            backbone = getattr(self.sam3_model, "backbone", None)
            if backbone is not None and hasattr(backbone, "forward_image"):
                return backbone.forward_image(x)
            if hasattr(self.sam3_model, "forward_image"):
                return self.sam3_model.forward_image(x)
        if hasattr(self.encoder, "forward_image"):
            return self.encoder.forward_image(x)
        return self.encoder(x)

    def _collect_4d_tensors(self, value) -> List[torch.Tensor]:
        """Collect spatial tensors from nested SAM3 outputs."""
        tensors = []
        if torch.is_tensor(value):
            if value.dim() == 4:
                tensors.append(value)
            return tensors
        if isinstance(value, dict):
            for nested in value.values():
                tensors.extend(self._collect_4d_tensors(nested))
            return tensors
        if isinstance(value, (list, tuple)):
            for nested in value:
                tensors.extend(self._collect_4d_tensors(nested))
        return tensors

    def _to_nchw(self, feature: torch.Tensor) -> torch.Tensor:
        """Normalize SAM3 spatial features to [B, C, H, W]."""
        if feature.dim() != 4:
            raise ValueError(f"Expected a 4D SAM3 feature map, got shape {tuple(feature.shape)}")
        if feature.shape[-1] > feature.shape[1] and feature.shape[1] == feature.shape[2]:
            feature = feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def _extract_sam3_spatial_feature(self, features) -> torch.Tensor:
        """Choose the SAM3 FPN level closest to the current 1/16 graph feature scale."""
        preferred_keys = [
            "backbone_fpn",
            "image_features",
            "vision_features",
            "fpn_features",
            "multi_scale_features",
            "sam2_backbone_out",
        ]
        candidates = []
        if isinstance(features, dict):
            for key in preferred_keys:
                if key in features:
                    candidates.extend(self._collect_4d_tensors(features[key]))
            if not candidates:
                candidates.extend(self._collect_4d_tensors(features))
        else:
            candidates.extend(self._collect_4d_tensors(features))
        if not candidates:
            raise RuntimeError(
                "Could not find a 4D spatial feature map in SAM3 output. "
                "Run a SAM3 smoke test and inspect backbone.forward_image() keys/shapes."
            )

        target_size = max(1, self.image_size // 16)

        def score(tensor: torch.Tensor) -> int:
            tensor = self._to_nchw(tensor)
            return abs(tensor.shape[-2] - target_size) + abs(tensor.shape[-1] - target_size)

        return self._to_nchw(min(candidates, key=score)).float()

    def _adapt_sam3_feature_map(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project SAM3 features to 256 channels and resize to the expected 1/16 grid."""
        if features.shape[1] != self._feature_dim:
            projection_key = str(features.shape[1])
            if projection_key not in self.sam3_feature_projections:
                raise RuntimeError(
                    f"SAM3 produced {features.shape[1]} channels, but this wrapper only has "
                    "predefined 1x1 projections for common channel sizes. Add a projection "
                    "for this channel count in SAM2Encoder.sam3_feature_projections."
                )
            projection = self.sam3_feature_projections[projection_key]
            features = projection(features)

        target_hw = (max(1, x.shape[-2] // 16), max(1, x.shape[-1] // 16))
        if features.shape[-2:] != target_hw:
            features = F.interpolate(features, size=target_hw, mode="bilinear", align_corners=False)
        return features
    
    def get_feature_scales(self) -> Dict[str, int]:
        """
        Get feature scales at different resolutions.
        
        Returns:
            Dictionary mapping scale names to resolution factors
        """
        # SAM encoder outputs at 1/16 scale
        # We can extract features at different scales from intermediate layers
        return {
            '1/4': 4,   # 128x128 for 512x512 input
            '1/8': 8,   # 64x64 for 512x512 input
            '1/16': 16, # 32x32 for 512x512 input (encoder output)
            '1/32': 32, # 16x16 for 512x512 input
        }
    
    @property
    def feature_dim(self) -> int:
        """
        Get feature dimension.
        
        Note: SAM encoder outputs 256 channels after the neck projection,
        regardless of the internal ViT dimension.
        """
        return self._feature_dim  # SAM encoder outputs 256 channels via neck projection

