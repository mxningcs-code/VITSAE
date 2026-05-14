# ViT SAE Notebook Pipeline

This notebook shows how a sparse autoencoder (SAE) is trained on internal
activations from a pretrained ViT-B/16 model, then used to inspect what the
learned SAE features respond to.

## Pipeline

1. **Load ImageNet validation images**
   - Read images from `data/val.X`.
   - Apply the official `ViT_B_16_Weights.IMAGENET1K_V1` preprocessing.
   - Each image is resized and normalized into a `[3, 224, 224]` tensor.

2. **Run pretrained ViT-B/16**
   - Load `torchvision.models.vit_b_16`.
   - Run a normal forward pass as a sanity check.
   - Use ImageNet top-5 predictions only to confirm the model works.

3. **Extract transformer activations**
   - Register a forward hook on selected ViT encoder layers.
   - Layer 6 uses `model.encoder.layers[5]`.
   - Layer 12 uses `model.encoder.layers[11]`.
   - The hooked activation shape is `[batch, 197, 768]`.
   - Token 0 is the CLS token; tokens 1-196 are patch tokens.

4. **Build SAE training data**
   - For patch-token SAEs:
     - Remove the CLS token.
     - Reshape `[images, 196, 768]` into `[images * 196, 768]`.
     - Each row is one image patch activation.
   - For CLS-token SAEs:
     - Keep only token 0.
     - Use `[images, 768]` as image-level activations.
   - Save activations and metadata under `outputs/activations`.

5. **Train the sparse autoencoder**
   - The SAE is a small two-layer model:

     ```python
     encoder = Linear(768, d_sae)
     z = ReLU(encoder(x))
     decoder = Linear(d_sae, 768)
     x_hat = decoder(z)
     ```

   - The training loss is:

     ```text
     total_loss = reconstruction_loss + l1_coeff * sparsity_loss
     reconstruction_loss = MSE(x_hat, x)
     sparsity_loss = mean(abs(z))
     ```

   - The reconstruction term makes the SAE preserve the original ViT
     activation.
   - The L1 sparsity term encourages each activation to use only a small
     number of SAE features.
   - Trained checkpoints and metrics are saved under
     `outputs/sae_checkpoints`.

6. **Rank SAE features**
   - Reload a trained SAE checkpoint.
   - Compute feature activations with `z = ReLU(sae.encoder(acts))`.
   - For each SAE feature, find the highest activating examples.
   - Patch-token features are ranked by top activating image patches.
   - CLS-token features are ranked by top activating full images.
   - Rankings are saved under `outputs/feature_rankings`.

7. **Visualize and interpret features**
   - Patch-token features are visualized as cropped patches and full images
     with red boxes around the activating patch.
   - CLS-token features are visualized as top activating full images.
   - Class purity is computed for CLS features to check whether a feature is
     category-related.
   - Visualizations are saved under `outputs/feature_visualizations`.

## Current Experiments

- `layer6_patch_sae`: trained on layer 6 patch-token activations.
- `layer12_patch_sae`: trained on layer 12 patch-token activations.
- `layer12_cls_sae`: trained on layer 12 CLS-token activations.

The patch-token SAEs mostly find local visual patterns such as background,
color, texture, and water-like regions. The CLS-token SAE is more image-level
and can produce more category-related features.

## Main Implementation Idea

The SAE is not part of the ViT model itself. The ViT is frozen and only used to
produce activations. The SAE is trained afterward as a separate model that tries
to reconstruct those activations using a sparse hidden representation. Each
hidden dimension of the SAE is treated as a candidate interpretable feature.
