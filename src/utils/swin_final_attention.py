import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


class SwinFinalStageAttention:
    """
    Correct attention extractor for timm Swin V2.
    Hooks the *softmax output* of the final WindowAttention block.
    """

    def __init__(self, model):
        self.model = model
        self.attn = None
        self.hook = None
        self._register_hook()

    def _register_hook(self):
        target_name = "backbone.layers.3.blocks.1.attn.softmax"

        for name, module in self.model.named_modules():
            if name == target_name:
                self.hook = module.register_forward_hook(self._hook_fn)
                print(f"Hook registered on: {name}")
                return

        raise RuntimeError("Could not find final Swin attention softmax layer.")

    def _hook_fn(self, module, input, output):
        # Expected shape: [B * num_windows, heads, N, N]
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            self.attn = output.detach()

    def remove(self):
        if self.hook is not None:
            self.hook.remove()

    def get_attention(self):
        if self.attn is None:
            raise RuntimeError("Attention tensor was never captured.")
        return self.attn


def visualize_swin_final_attention(
    model,
    image_tensor,
    save_path,
    target_size=256
):
    """
    Paper-grade Swin V2 final-stage attention visualization (timm-compatible).
    """

    device = image_tensor.device
    model.eval()

    extractor = SwinFinalStageAttention(model)

    with torch.no_grad():
        _ = model(image_tensor)

    attn = extractor.get_attention()  # [B * num_windows, heads, N, N]
    extractor.remove()

    # Average over heads
    attn = attn.mean(dim=1)  # [B * num_windows, N, N]

    # Average over windows to get global token relevance
    attn = attn.mean(dim=0)  # [N, N]

    attn = attn + torch.eye(attn.size(-1), device=attn.device)
    attn = attn / attn.sum(dim=-1, keepdim=True)

    relevance = attn.mean(dim=0)  # [N]

    N = relevance.shape[0]
    spatial = int(np.sqrt(N))

    if spatial * spatial != N:
        raise RuntimeError(f"Final stage token count {N} is not square.")

    heatmap = relevance.reshape(spatial, spatial).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap, (target_size, target_size))

    img = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    heat_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = 0.6 * heat_color[..., ::-1] / 255.0 + 0.4 * img

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Swin V2 Final Attention")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
