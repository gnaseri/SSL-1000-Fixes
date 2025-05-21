# SSL-1000-Fixes
Suggested Next Steps for 1000 Common Issues in Self-Supervised Learning

---

## 1. Representation Rank is Low

**Symptoms:**
- Low covariance matrix rank or eigenvalue entropy.

**Suggested Next Steps:**
- Increase the variance loss coefficient (e.g., in VICReg).
- Increase the embedding dimension (if too narrow).
- Increase batch size to capture more variation.
- Use stronger augmentations to diversify views.
- Add residual connections to encourage diversity in output.

---

## 2. Learned Kernels Are Noisy or Dead

**Symptoms:**
- Kernels look random or many are close to zero.

**Suggested Next Steps:**
- Lower the learning rate or add gradient clipping.
- Apply weight decay or L1 regularization on conv weights.
- Inspect data normalization — ensure it’s not too aggressive.
- Add initialization schemes like Kaiming or orthogonal init.

---

## 3. Embeddings Collapse or Are Too Similar

**Symptoms:**
- Cosine similarity between features too high, low variance.

**Suggested Next Steps:**
- Add or strengthen variance regularization.
- Introduce weak asymmetry (e.g., stop gradient, predictor MLP).
- Delay EMA updates if using target networks.
- Increase prediction horizon to reduce trivial matching.

---

## 4. Early Overfitting or Poor Generalization

**Symptoms:**
- Training loss drops, but validation loss rises quickly.

**Suggested Next Steps:**
- Add dropout, augmentations, or weight decay.
- Check if the model is memorizing shortcuts (like pixel positions).
- Use early stopping or learning rate warmup.
- Reduce model capacity or simplify the task.

---

## 5. Feature Norms Are Highly Uneven

**Symptoms:**
- Some features dominate (e.g., 2–3 features have high norms).

**Suggested Next Steps:**
- Normalize features before loss computation.
- Strengthen covariance decorrelation loss.
- Use BatchNorm or LayerNorm in the projector or encoder.
- Consider whitening transforms in analysis.

---

## 6. Covariance Matrix is Highly Correlated

**Symptoms:**
- High off-diagonal energy, many features redundant.

**Suggested Next Steps:**
- Increase covariance loss coefficient (e.g., VICReg’s nu).
- Reduce use of overlapping augmentations that induce similarity.
- Increase depth or non-linearity in encoder/projector.

---

## 7. Eigenvalues Decay Too Quickly

**Symptoms:**
- Only top 5–10 components explain most variance.

**Suggested Next Steps:**
- Increase batch diversity or apply feature decorrelation.
- Try orthogonality constraints (e.g., QR loss or mutual information maximization).
- Apply contrastive loss with negative samples.

---

## 8. Prediction Errors Are Localized or Systematic

**Symptoms:**
- Specific regions or classes consistently fail.

**Suggested Next Steps:**
- Analyze dataset imbalance or sample quality.
- Add local receptive field diversity (e.g., dilated convolutions).
- Train with curriculum or targeted augmentation on weak regions.

---

## 9. Latent Space Visualization Shows Poor Separation

**Symptoms:**
- t-SNE or PCA shows all points clumped together.

**Suggested Next Steps:**
- Improve augmentation diversity to avoid trivial alignment.
- Add auxiliary tasks (e.g., rotation prediction or jigsaw).
- Increase embedding dimensionality or training time.

---

## 10. Gradient Norms Are Exploding or Vanishing

**Symptoms:**
- Training becomes unstable, NaNs in loss, or gradients ~0.

**Suggested Next Steps:**
- Use gradient clipping (e.g., clip value or clip norm).
- Check for ReLU dead units — try LeakyReLU or GELU.
- Switch to Adam or RMSprop with proper eps setting.
- Use smaller learning rate or mixed-precision training with GradScaler.

---

## 11. Loss Plateaus Early

**Symptoms:**
- Loss stops improving in early epochs.

**Suggested Next Steps:**
- Use a learning rate scheduler (StepLR, CosineAnnealing, etc.).
- Add warmup steps at the start.
- Increase model capacity (width or depth).
- Check for data leakage or label noise if applicable.

---

## 12. Very Low Output Variance

**Symptoms:**
- All output features are near zero or constant across samples.

**Suggested Next Steps:**
- Initialize final layer biases to small non-zero values.
- Use LayerNorm in the output head.
- Regularize output to have unit variance.
- Inspect input normalization and activation function saturation.

---

## 13. Poor Transfer to Downstream Task

**Symptoms:**
- Learned representations perform poorly on supervised tasks.

**Suggested Next Steps:**
- Introduce task-aware pretext signals.
- Add semantic consistency loss if applicable.
- Use larger datasets or pretrained backbones.
- Fine-tune with a small supervised head to guide representation.

---

## 14. Slow Training

**Symptoms:**
- Training epochs take too long.

**Suggested Next Steps:**
- Optimize data loading (e.g., increase num_workers, use .pin_memory()).
- Use mixed precision or channels_last format.
- Profile with torch.profiler or nvprof to locate bottlenecks.
- Chunk long sequences if you’re doing sequential modeling.

---

## 15. BatchNorm or LayerNorm Acting Erratically

**Symptoms:**
- Normalization hurts performance or destabilizes training.

**Suggested Next Steps:**
- Try GroupNorm or InstanceNorm instead.
- Replace with no normalization and tune learning rate carefully.
- If using DDP: ensure BatchNorm is synchronized (SyncBatchNorm).

---

## 16. Representations Are Nearly Identical Across Augmented Views

**Symptoms:**
- High cosine similarity between outputs from different views.

**Suggested Next Steps:**
- Add stochastic layers (e.g., dropout) before the projector.
- Increase augmentation intensity (e.g., stronger color jitter, masking).
- Add temporal or spatial shift if applicable (e.g., for video or sequence data).

---

## 17. Checkpoint Sizes Are Too Large

**Symptoms:**
- Disk fills up quickly or model is hard to share.

**Suggested Next Steps:**
- Save only model weights, not full optimizer states.
- Use state_dict() filtering to exclude unneeded buffers.
- Save in float16 if inference is not affected.

---

## 18. Embedding Dimensions Used Unevenly

**Symptoms:**
- Some dimensions are always close to 0, others dominate.

**Suggested Next Steps:**
- Add uniformity loss or entropy maximization on features.
- Apply decorrelation or orthogonality constraints.
- Use learned positional encoding or non-linear projectors.

---

## 19. EMA Target Doesn’t Help

**Symptoms:**
- EMA-targeted representations don’t stabilize or improve.

**Suggested Next Steps:**
- Reduce the EMA decay at early epochs (e.g., start at 0.9 and anneal to 0.99).
- Re-initialize EMA with online model every few epochs (experimental).
- Try using fixed encoder instead of EMA to test isolation.

---

## 20. Visualization Metrics Don’t Change Over Epochs

**Symptoms:**
- PCA, cosine similarity, rank, or eigenvalue plots are flat.

**Suggested Next Steps:**
- Increase the training duration.
- Visualize per-class metrics or region-specific patterns.
- Log metrics after more granular intervals (e.g., every few batches).
- Inspect if loss is still changing — maybe the model is learning but not diversifying representations.

---

## 21. Feature Activations Are Sparse or Mostly Zero

**Symptoms:**
- Most features in intermediate layers are zero.

**Suggested Next Steps:**
- Replace ReLU with GELU, ELU, or Swish to allow negative flows.
- Add BatchNorm/LayerNorm before nonlinearity to stabilize activation spread.
- Lower the weight decay, which might be shrinking weights excessively.

---

## 22. Learned Representations Are Overly Sensitive to Noise

**Symptoms:**
- Small input perturbations drastically change output.

**Suggested Next Steps:**
- Add consistency loss (e.g., L2 between outputs of noisy inputs).
- Use adversarial training or noise augmentations.
- Try input denoising autoencoder or robust contrastive objectives.

---

## 23. Training Is Stochastic or Unstable Across Runs

**Symptoms:**
- Large performance variation between seeds.

**Suggested Next Steps:**
- Set all seeds (torch, numpy, random) and enable `torch.backends.cudnn.deterministic = True`.
- Use larger batch sizes or gradient accumulation.
- Tune EMA decay and optimizer momentum for stability.

---

## 24. Model Predicts Only the Majority Class or Background

**Symptoms:**
- Class imbalance in predictions.

**Suggested Next Steps:**
- Use class-balanced sampling or weighted loss functions.
- Over-sample minority classes or under-sample majority ones.
- Add focal loss or label smoothing to improve calibration.

---

## 25. Model Performs Well Quantitatively but Fails Visually

**Symptoms:**
- High accuracy but poor qualitative results (e.g., in reconstructions or predictions).

**Suggested Next Steps:**
- Add perceptual loss or structural similarity index (SSIM).
- Visualize feature maps or intermediate outputs to detect bottlenecks.
- Investigate if model is overfitting shortcut features.

---

## 26. High Feature Redundancy Across Time or Space

**Symptoms:**
- Features don’t evolve across spatial/temporal positions.

**Suggested Next Steps:**
- Use multi-scale features or dilated convolutions.
- Encourage diversity with contrastive prediction across time or space.
- Add shuffle-and-predict or jigsaw tasks to break local patterns.

---

## 27. Learned Embeddings Show Bias (e.g., class, texture, position)

**Symptoms:**
- Embeddings cluster by confounding factors.

**Suggested Next Steps:**
- Add domain adversarial training to suppress bias signals.
- Use class-conditioned augmentation or stratified sampling.
- Regularize with mutual information minimization between embeddings and bias.

---

## 28. Final Layer Activations Are Saturated

**Symptoms:**
- Activations stuck near ±1 (e.g., tanh or sigmoid saturation).

**Suggested Next Steps:**
- Use identity or linear activation in last layer for representation learning.
- Normalize final layer outputs before loss (e.g., L2 norm for cosine similarity).
- Replace sigmoid/tanh with scaled versions or avoid them altogether.

---

## 29. Latent Space Changes Are Too Abrupt Between Epochs

**Symptoms:**
- Drastic shift in embedding space across epochs.

**Suggested Next Steps:**
- Reduce learning rate or use cosine annealing.
- Stabilize training with EMA targets or slower scheduler decay.
- Visualize trajectory of embeddings to identify jump points.

---

## 30. Too Many Metrics Make it Hard to Interpret

**Symptoms:**
- Overwhelming WandB or CSV logs with unclear insight.

**Suggested Next Steps:**
- Group metrics into categories (e.g., loss/, rep/, kernels/).
- Plot composite metrics: entropy, rank, norm, diversity.
- Use EMA smoothing or epoch-wise summaries.

---

## 31. Feature Norms Grow Over Time

**Symptoms:**
- Feature magnitudes increase steadily, possibly causing instability.

**Suggested Next Steps:**
- Apply feature normalization (e.g., L2) before computing loss.
- Add a feature norm penalty to encourage compactness.
- Use BatchNorm or LayerNorm in deeper layers.

---

## 32. Learned Representations Are Too Task-Specific

**Symptoms:**
- Representations don’t transfer well or generalize poorly to new domains.

**Suggested Next Steps:**
- Use multi-task learning or domain generalization objectives.
- Apply domain-specific augmentations during pretraining.
- Encourage invariance to domain-specific features with auxiliary losses.

---

## 33. Loss Has High Variance Across Batches

**Symptoms:**
- Spiky loss curve within epochs.

**Suggested Next Steps:**
- Increase batch size or enable gradient accumulation.
- Use gradient clipping or smoother optimizers (e.g., RMSprop).
- Check for label noise, outliers, or data corruption.

---

## 34. Model Learns Trivial Identity Mapping

**Symptoms:**
- Autoencoder or predictor learns to copy input.

**Suggested Next Steps:**
- Increase prediction difficulty (e.g., masked inputs, farther temporal targets).
- Add bottlenecks (e.g., smaller latent size, dropout).
- Use contrastive or predictive coding objectives.

---

## 35. Cosine Similarities Between Embeddings Are All ~1

**Symptoms:**
- Representations collapse toward the same direction.

**Suggested Next Steps:**
- Introduce decorrelation loss (as in VICReg, Barlow Twins).
- Verify augmentations are not too weak (causing alignment to dominate).
- Add predictor asymmetry or stop-gradient operations.

---

## 36. Augmentations Make Model Too Invariant

**Symptoms:**
- Loss increases or performance drops after aggressive augmentation.

**Suggested Next Steps:**
- Tune augmentation intensity based on dataset.
- Use augmentation mixing (e.g., random select from strong/weak).
- Include augmentation-specific loss components (e.g., patch-wise consistency).

---

## 37. Temporal Representations Are Not Predictive

**Symptoms:**
- Predictive performance on future states is poor.

**Suggested Next Steps:**
- Add positional encoding or temporal convolutions.
- Use causal attention or masked convolutions for temporal structure.
- Predict multiple future steps instead of just one.

---

## 38. Spatial Attention Maps Are Too Uniform

**Symptoms:**
- No distinct focus regions in visual attention models.

**Suggested Next Steps:**
- Add attention diversity loss.
- Reduce model’s global pooling dependence.
- Use local self-attention or hierarchical attention blocks.

---

## 39. Embeddings Fail to Capture Class Separability

**Symptoms:**
- No clustering in embedding space despite supervised signals.

**Suggested Next Steps:**
- Add contrastive loss or triplet loss during supervised phase.
- Use center loss or ArcFace-style margins to pull apart classes.
- Visualize with t-SNE or UMAP to better interpret overlaps.

---

## 40. Embedding Space Is Not Smooth

**Symptoms:**
- Small changes in input lead to large jumps in embedding.

**Suggested Next Steps:**
- Apply Lipschitz regularization or Jacobian norm penalty.
- Use interpolation consistency training (e.g., MixUp, CutMix).
- Add a manifold smoothing loss between nearby embeddings.

---

## 41. Training Works but Validation Always Lags Behind

**Symptoms:**
- Validation performance never catches up to training.

**Suggested Next Steps:**
- Increase data augmentation or use test-time augmentation.
- Inspect for distribution shift between train/val datasets.
- Try label smoothing or confidence calibration losses.

---

## 42. Model Too Slow During Inference

**Symptoms:**
- Good training speed but slow forward pass at test time.

**Suggested Next Steps:**
- Replace inefficient layers (e.g., use depthwise separable convs).
- Use TorchScript, ONNX, or TensorRT to optimize inference.
- Prune or quantize model using PyTorch Quantization Toolkit.

---

## 43. Early Epochs Improve, Then Suddenly Diverge

**Symptoms:**
- Model trains initially then becomes unstable.

**Suggested Next Steps:**
- Reduce learning rate decay speed.
- Enable gradient norm tracking — add alerts for spikes.
- Monitor activation magnitudes to catch exploding outputs.

---

## 44. Loss Curve Shows Sudden Jumps or Drops

**Symptoms:**
- Sharp spikes in training/validation loss.

**Suggested Next Steps:**
- Check for batch corruption or augmentation bugs.
- Add gradient clipping to prevent numerical overflow.
- Re-initialize suspicious layers if only one head is unstable.

---

## 45. PCA Shows Linear Patterns, Not Clusters

**Symptoms:**
- Projected embeddings lie on a line or plane.

**Suggested Next Steps:**
- Increase embedding dimensionality or batch size.
- Add non-linearities or residual connections in the projector.
- Use multi-view losses (e.g., more than 2 views in contrastive learning).

---

## 46. Learned Filters Are Identical or Repeating

**Symptoms:**
- Several convolution filters are near-duplicates.

**Suggested Next Steps:**
- Add orthogonality regularization on convolutional weights.
- Use group convolutions or increase kernel diversity via dropout.
- Enforce low-rank decomposition or sparsity constraints.

---

## 47. Positional Encoding Seems Ineffective

**Symptoms:**
- Model ignores order or position in sequences.

**Suggested Next Steps:**
- Switch from sinusoidal to learned positional embeddings.
- Use relative position encoding (as in Transformer-XL).
- Add position prediction tasks to reinforce order awareness.

---

## 48. Embedding Norms Vary Greatly Across Batches

**Symptoms:**
- Some batches have much higher norms than others.

**Suggested Next Steps:**
- Add batch-wise norm regularization.
- Normalize at the feature level, not the batch level, before projection.
- Log norm histogram across multiple batches and average.

---

## 49. Feature Collapse Only Happens in One Branch

**Symptoms:**
- One encoder or projection path dominates.

**Suggested Next Steps:**
- Tune EMA decay or learning rates independently for branches.
- Add balance loss between online and target representations.
- Ensure gradient is not leaking through stop-gradient nodes.

---

## 50. Model Is Too Sensitive to Hyperparameters

**Symptoms:**
- Tiny hyperparameter changes cause big performance shifts.

**Suggested Next Steps:**
- Add regularization to reduce sensitivity.
- Use hyperparameter sweeps with logging (e.g., Optuna + WandB).
- Start with robust optimizers (AdamW, Lookahead, Lion).

---

## 51. Features Are Disentangled but Not Predictive

**Symptoms:**
- Latent dimensions are cleanly separated but don’t correlate with useful downstream variables.

**Suggested Next Steps:**
- Incorporate supervised fine-tuning on downstream tasks.
- Add mutual information maximization between features and labels.
- Apply causal feature selection to determine which representations matter.

---

## 52. Training Becomes Slower Over Time

**Symptoms:**
- Epoch time increases as training progresses.

**Suggested Next Steps:**
- Investigate data loader memory leaks or GPU fragmentation.
- Restart the job periodically or clear CUDA cache manually.
- Profile with torch.profiler to see if backward pass is bloating.

---

## 53. Final Layer Has Very High or Low Weight Norm

**Symptoms:**
- Final linear/projection layer absorbs too much or too little weight magnitude.

**Suggested Next Steps:**
- Apply weight norm penalty or L2 normalization on output.
- Consider weight normalization or spectral normalization.
- Use layer scaling (small init gain) to control output range.

---

## 54. Representations Work Only With Linear Probes

**Symptoms:**
- Linear classifiers perform well but non-linear downstream tasks fail.

**Suggested Next Steps:**
- Enrich pretraining with nonlinear objectives or auxiliary heads.
- Try representation fusion across layers (e.g., skip connections).
- Add nonlinear probes during analysis to reveal hidden structure.

---

## 55. Model Learns Shortcut Artifacts (e.g., borders, color)

**Symptoms:**
- Model overfits visual or statistical artifacts instead of semantic signals.

**Suggested Next Steps:**
- Use background substitution or style augmentation.
- Add patch mixing or region swapping.
- Train with adversarial examples to suppress shortcut reliance.

---

## 56. Highly Entangled Representations

**Symptoms:**
- Each dimension mixes multiple factors (e.g., pitch + rhythm in audio).

**Suggested Next Steps:**
- Apply factorized loss (e.g., predict rhythm separately).
- Use β-VAE, infoGAN, or total correlation minimization.
- Introduce structured bottlenecks to force separation.

---

## 57. Strong Representations but Weak Predictions

**Symptoms:**
- Intermediate features are rich, but final outputs are weak.

**Suggested Next Steps:**
- Inspect predictor head capacity — may need more depth or nonlinearity.
- Use intermediate supervision (deep supervision).
- Try ensembling predictions across heads or layers.

---

## 58. Embeddings Are Invariant to Too Many Transformations

**Symptoms:**
- Model ignores differences that are actually important (e.g., time shift, expression, class).

**Suggested Next Steps:**
- Refine augmentations: make only task-irrelevant ones invariant.
- Use contrastive pairs with task-relevant variation.
- Add equivariance constraints instead of full invariance.

---

## 59. Loss Improves but Evaluation Metric Stalls

**Symptoms:**
- Loss function keeps decreasing, but downstream metrics (e.g., accuracy, F1, NMI) do not.

**Suggested Next Steps:**
- Double-check that loss correlates with the true objective.
- Use evaluation metrics as auxiliary loss terms.
- Replace MSE with contrastive or ranking losses that better reflect task needs.

---

## 60. Model Struggles With Out-of-Distribution (OOD) Inputs

**Symptoms:**
- Good performance on in-distribution data but fails on minor shifts.

**Suggested Next Steps:**
- Train with domain randomization or style transfer.
- Use Outlier Exposure or self-supervised OOD detection.
- Apply Bayesian or ensemble methods to estimate uncertainty.

---

## 61. Model Quickly Memorizes Training Set

**Symptoms:**
- Training accuracy reaches 100% rapidly, validation stagnates.

**Suggested Next Steps:**
- Increase data augmentation and randomization.
- Apply label smoothing to reduce overconfidence.
- Add dropout, cutout, or mixup to regularize learning.

---

## 62. Model Fails to Learn From Rare Examples

**Symptoms:**
- Minority class or edge-case inputs are consistently mispredicted.

**Suggested Next Steps:**
- Apply oversampling or loss reweighting (e.g., focal loss).
- Use curriculum learning to emphasize rare samples over time.
- Introduce contrastive pairs between rare and common instances.

---

## 63. Model’s Confidence Is Always High

**Symptoms:**
- Predicts with near-certain confidence, even when wrong.

**Suggested Next Steps:**
- Add confidence penalty or entropy maximization regularizer.
- Apply temperature scaling or Bayesian uncertainty estimation.
- Train with label noise or adversarial samples to soften confidence.

---

## 64. Representations Have High Mutual Information With Augmentations

**Symptoms:**
- Embeddings leak augmentation type (e.g., rotation, color jitter).

**Suggested Next Steps:**
- Add augmentation-invariance loss (e.g., contrastive across augment types).
- Mask augmentation identity from predictor.
- Include augmentation prediction task and subtract gradients (as in adversarial multitask setups).

---

## 65. Projection Head Is Bottlenecking Learning

**Symptoms:**
- Increasing projection head depth reduces performance.

**Suggested Next Steps:**
- Try skip connections across projection layers.
- Reduce bottleneck dimension only after sufficient training.
- Move regularization from projection head to encoder directly.

---

## 66. Layer Outputs Are Identical Across Samples

**Symptoms:**
- Intermediate activations are static for different inputs.

**Suggested Next Steps:**
- Use batch-dependent normalization (BatchNorm or GroupNorm).
- Replace static convolutions with dynamic or conditional layers.
- Check for frozen parameters or misinitialized layers.

---

## 67. Kernel Weights Are All Small in Magnitude

**Symptoms:**
- Kernels converge to very low norm, model becomes underpowered.

**Suggested Next Steps:**
- Add weight magnitude regularization (maximize norm).
- Remove or reduce L2 regularization if too aggressive.
- Track gradient norms per layer to ensure flow is active.

---

## 68. Representation Similarity Across Batches Is Too High

**Symptoms:**
- Features learned from different batches are overly aligned.

**Suggested Next Steps:**
- Add batch decorrelation loss.
- Use batch shuffling to reduce alignment bias.
- Compare representations across augment sets, not just batch samples.

---

## 69. Final Output Layer Overfits the Loss Function

**Symptoms:**
- Very low loss, but meaningless or misleading predictions.

**Suggested Next Steps:**
- Replace final linear layer with nonlinear or gated alternatives.
- Regularize final layer weights (e.g., spectral norm).
- Detach loss target partially (e.g., stop gradient halfway through).

---

## 70. Self-Supervised Learning Works Only on Specific Data Formats

**Symptoms:**
- SSL succeeds on spectrograms but fails on raw waveforms, etc.

**Suggested Next Steps:**
- Use multi-view SSL across formats (e.g., CQT + waveform).
- Add domain adapters to bridge modalities.
- Train with cross-modal prediction or co-training tasks.

---

## 71. Attention Maps Are Uniform or Random

**Symptoms:**
- Self-attention fails to focus on meaningful regions.

**Suggested Next Steps:**
- Add auxiliary attention supervision if labels available (e.g., saliency).
- Use local/global attention hybrids (e.g., Swin, deformable attention).
- Regularize attention maps with entropy loss or diversity loss across heads.

---

## 72. Output Depends Heavily on Positional Encoding

**Symptoms:**
- Removing positional encodings drastically drops performance.

**Suggested Next Steps:**
- Use learned relative positions instead of absolute.
- Introduce data augmentations that shuffle position to test robustness.
- Add position prediction as an auxiliary task.

---

## 73. Loss Oscillates in Later Epochs

**Symptoms:**
- Noisy up-down loss in late training.

**Suggested Next Steps:**
- Try lowering the learning rate or switch to a smoother optimizer.
- Use Poly decay or Cosine Annealing with restarts.
- Check for numerical instability or NaNs in gradients.

---

## 74. First Few Layers Are Frozen in Learning

**Symptoms:**
- Weights of early layers barely change.

**Suggested Next Steps:**
- Add layer-wise learning rates — increase early layer LR.
- Add skip connections to force gradient flow.
- Inspect input normalization or weight scale mismatch.

---

## 75. Embeddings Are Nearly Binary (Hard Clustering)

**Symptoms:**
- Output features are ~0 or ~1, like hard assignment.

**Suggested Next Steps:**
- Add entropy maximization or softmax temperature tuning.
- Introduce noise in the representation (e.g., Gaussian noise).
- Reduce use of sharpening functions (e.g., hard Gumbel-softmax).

---

## 76. Representations Vary Too Much Across Augmentations

**Symptoms:**
- Representations for the same image under augmentations are too dissimilar.

**Suggested Next Steps:**
- Add alignment loss (L2/cosine between augmented views).
- Reduce augmentation strength (especially when starting training).
- Use temporal averaging or EMA smoothing across augmentations.

---

## 77. Cosine Similarity Between Batches Is Too Low

**Symptoms:**
- Batches of features don’t align at all — model is unstructured.

**Suggested Next Steps:**
- Reduce model capacity to prevent random exploration.
- Add soft clustering loss or global contrastive anchors.
- Inspect for data shuffling bugs causing semantic mismatch.

---

## 78. Model Stops Responding to New Data

**Symptoms:**
- Introducing new data doesn’t affect loss or features.

**Suggested Next Steps:**
- Check for batch normalization stats — may have saturated.
- Increase learning rate temporarily to overcome flat region.
- Reset optimizer state or remove stale EMA tracking.

---

## 79. KL Divergence or InfoNCE Loss is Always Near Zero

**Symptoms:**
- Representation space is uniform or trivially aligned.

**Suggested Next Steps:**
- Reduce use of strong priors (e.g., strong regularization or layer norm).
- Add negative sampling diversity (e.g., memory banks or cross-batch).
- Add temperature tuning in softmax.

---

## 80. Model Performance Degrades After Pretraining

**Symptoms:**
- Transfer performance is worse than random init.

**Suggested Next Steps:**
- Pretrain with task-relevant augmentations or sampling.
- Check if encoder is overfitted to self-supervised proxy task.
- Try partial freezing or reinitializing the final layers during finetuning.

---

## 81. BatchNorm Statistics Drift Wildly Between Epochs

**Symptoms:**
- Sudden jumps in feature norms or accuracy.

**Suggested Next Steps:**
- Replace with GroupNorm or LayerNorm for more stability.
- Switch to SyncBatchNorm in multi-GPU setups.
- Reduce batch size but accumulate stats across steps manually.

---

## 82. PCA Shows Embedding Collapse Only in One Class

**Symptoms:**
- One class has low-variance collapsed representations.

**Suggested Next Steps:**
- Apply class-conditional VC regularization (only penalize collapse inside each class).
- Introduce class-aware augmentations.
- Visualize feature spread per class and adjust sampling accordingly.

---

## 83. Loss Minimizes Too Fast in First Epoch

**Symptoms:**
- Sudden drop in loss without generalization gain.

**Suggested Next Steps:**
- Increase warmup steps for optimizer.
- Try cyclical learning rate to recover from bad convergence basin.
- Track gradient norms across layers — signs of early saturation.

---

## 84. Gradients Are Nonzero but Do Not Change Parameters

**Symptoms:**
- Model reports active gradients, but weights stay almost unchanged.

**Suggested Next Steps:**
- Check for gradient clipping too tight (e.g., clip_value=0.1).
- Inspect optimizer momentum or weight decay conflicts.
- Log parameter delta norms per update to confirm actual change.

---

## 85. Same Features Are Activated Across All Classes

**Symptoms:**
- Activations in final or intermediate layers are class-agnostic.

**Suggested Next Steps:**
- Introduce class-specific loss components (e.g., center loss, ArcFace).
- Add conditional projection heads.
- Visualize per-class activation heatmaps to detect overlaps.

---

## 86. Model Fails After Moving to Mixed Precision

**Symptoms:**
- Loss explodes or underflows when using AMP.

**Suggested Next Steps:**
- Inspect for instabilities in small-scale activations.
- Use GradScaler() properly and log scale overflow events.
- Temporarily disable AMP on sensitive layers (e.g., batch norm).

---

## 87. Outputs Saturate at Specific Values (e.g., 0 or 1)

**Symptoms:**
- Model output distribution collapses to a small range.

**Suggested Next Steps:**
- Remove final activation function if unnecessary (e.g., sigmoid before contrastive loss).
- Check for label encoding mismatch (e.g., float vs int targets).
- Clip or rescale labels if they cause saturation in early loss curves.

---

## 88. Per-layer Gradient Norms Are Inverted

**Symptoms:**
- Downstream layers receive stronger updates than upstream.

**Suggested Next Steps:**
- Use grad norm balancing or gradient scaling hooks.
- Add skip connections to improve gradient flow.
- Consider progressive layer unfreezing if using transfer learning.

---

## 89. Self-Supervised Representations Are Too Fine-Grained

**Symptoms:**
- Model overfocuses on textures or local details, underperforms on generalization.

**Suggested Next Steps:**
- Apply style randomization or texture-blind augmentations.
- Include global context prediction tasks.
- Penalize local redundancy using contrastive patch-to-global loss.

---

## 90. Early Layers Dominate Kernel Norms

**Symptoms:**
- Weight energy concentrates in first 1–2 conv layers.

**Suggested Next Steps:**
- Normalize all kernel norms per layer.
- Add layer-wise weight regularization to flatten energy profile.
- Visualize kernel statistics to track norm growth over time.

---

## 91. Weight Norms Grow Without Bound

**Symptoms:**
- Unchecked weight magnitude increases even with good validation metrics.

**Suggested Next Steps:**
- Use weight decay, trust region methods, or adaptive optimizers.
- Add explicit norm constraint loss (e.g., L2 penalty > target threshold).
- Normalize weights before output projection.

---

## 92. Similar Input Samples Lead to Divergent Outputs

**Symptoms:**
- Small variations in inputs cause inconsistent predictions.

**Suggested Next Steps:**
- Add consistency regularization (e.g., Mean Teacher or R-Drop).
- Apply feature noise injection and penalize output divergence.
- Check if input preprocessing is unstable (e.g., resizing artifacts).

---

## 93. Feature Channels Are Imbalanced

**Symptoms:**
- Some feature channels are always high or dead across all data.

**Suggested Next Steps:**
- Visualize channel-wise statistics (mean, std, activation frequency).
- Apply channel dropout or learnable channel gating.
- Add channel-wise decorrelation loss (e.g., as in Barlow Twins).

---

## 94. Augmented Views Are Too Predictable

**Symptoms:**
- Different augmentations of the same input lead to nearly identical representations.

**Suggested Next Steps:**
- Increase augmentation diversity and randomness (e.g., strong+weak view pairing).
- Add augmentation-type prediction as a pretext task.
- Use ViewDrop, PatchDrop, or other masking-style augmentations to promote unpredictability.

---

## 95. Model Learns Slowly Despite Good Architecture

**Symptoms:**
- Loss decreases very slowly even with powerful architectures.

**Suggested Next Steps:**
- Warm up with easier proxy tasks (e.g., autoencoding or next-patch prediction).
- Apply layer-wise adaptive learning rates (e.g., LARS or Ranger).
- Check that gradients are not being silently clipped or nan-ed.

---

## 96. Training Collapses When Switching Dataset

**Symptoms:**
- Model works on one dataset but diverges or collapses on another.

**Suggested Next Steps:**
- Check if input normalization statistics need adjusting.
- Log label distributions, length distributions, or input ranges.
- Pretrain on the new dataset using a simplified task, then fine-tune.

---

## 97. Representations Are Not Smooth in Latent Space

**Symptoms:**
- Interpolating between embeddings leads to unnatural outputs.

**Suggested Next Steps:**
- Use variational bottlenecks (VAE-style).
- Add latent interpolation consistency loss.
- Train with interpolation-based contrastive learning (e.g., MixCo, InterCLR).

---

## 98. Training Requires Frequent Restarts

**Symptoms:**
- Model gets stuck in poor minima unless manually restarted.

**Suggested Next Steps:**
- Use cyclical learning rates or random restarts as part of training.
- Consider sharpness-aware minimization (SAM) to flatten loss landscape.
- Add noise injection to weights or activations during training.

---

## 99. Feature Collapse Only Happens in Subgroups

**Symptoms:**
- Certain classes, time segments, or data sources show low-rank features.

**Suggested Next Steps:**
- Apply group-wise VC regularization (e.g., per class, per domain).
- Visualize feature diversity conditioned on subgroup identity.
- Use mixture-of-experts or domain-specific branches.

---

## 100. Latent Variables Show Strong Mutual Redundancy

**Symptoms:**
- High mutual information between feature dimensions.

**Suggested Next Steps:**
- Apply total correlation regularization (e.g., in β-TCVAE).
- Use PCA whitening or Barlow Twins-style decorrelation loss.
- Introduce structured sparsity to push diversity across features.

---

## 101. Encoder Layers Are Bypassed

**Symptoms:**
- Final features ignore early-layer representations.

**Suggested Next Steps:**
- Add skip connections and supervise intermediate outputs.
- Penalize activation redundancy across layers.
- Replace deep linear chains with multi-scale feature fusion.

---

## 102. Noise Sensitivity Appears Only During Inference

**Symptoms:**
- Model behaves robustly during training but is brittle at test time.

**Suggested Next Steps:**
- Add test-time augmentation (e.g., augment-and-aggregate).
- Perform Monte Carlo dropout or ensembling during inference.
- Train with distributional robustness objectives.

---

## 103. Positional Encodings Dominate Early Training

**Symptoms:**
- Position information overwhelms content features.

**Suggested Next Steps:**
- Decay position encodings over time (e.g., annealed strength).
- Switch to relative position encoding.
- Add auxiliary tasks that rely solely on content, not position.

---

## 104. Model Can’t Differentiate Easy from Hard Samples

**Symptoms:**
- Accuracy is uniform across all sample types (easy/hard, frequent/rare).

**Suggested Next Steps:**
- Add difficulty-aware losses (e.g., focal loss, margin-based reweighting).
- Use self-paced learning or curriculum learning.
- Label easy vs. hard examples with a confidence head.

---

## 105. Model Learns Dataset-Specific Artifacts

**Symptoms:**
- Performs well on training set, but fails on slightly restructured data (e.g., shifted, rescaled, renamed classes).

**Suggested Next Steps:**
- Augment with synthetic transformations or adversarial corruption.
- Apply domain randomization (e.g., color, blur, resolution).
- Add artifact suppression loss (e.g., decorrelate background).

---

## 106. Embedding Similarity Is Not Transitive

**Symptoms:**
- A ≈ B and B ≈ C, but A ≠ C in latent space.

**Suggested Next Steps:**
- Apply triplet loss or relational contrastive loss to enforce transitivity.
- Introduce semantic anchors (e.g., prototype centers).
- Try contrastive clustering frameworks (e.g., SwAV, DeepCluster).

---

## 107. Model Improves Accuracy but Forgets Earlier Examples

**Symptoms:**
- Accuracy goes up overall but drops on previously mastered samples.

**Suggested Next Steps:**
- Use Elastic Weight Consolidation (EWC) or LwF to retain past knowledge.
- Add memory replay buffer for prior batch samples.
- Introduce importance-weighted regularization on older activations.

---

## 108. Features Are Over-Aligned Across Classes

**Symptoms:**
- Representations for different classes fall into same regions.

**Suggested Next Steps:**
- Add class-conditional contrastive terms.
- Apply Supervised Contrastive Learning (SupCon).
- Add margin penalties between class centers.

---

## 109. Model Fails to Learn Abstract Patterns

**Symptoms:**
- Learns pixel-level or token-level structure, but not higher-level concepts.

**Suggested Next Steps:**
- Add hierarchical prediction objectives (e.g., segment-level prediction).
- Use multi-level pooling and cross-scale attention.
- Introduce task-aware augmentation (e.g., mask shapes, object removal).

---

## 110. Latent Space Is Not Interpretable

**Symptoms:**
- Difficult to associate any axis or region with meaningful behavior.

**Suggested Next Steps:**
- Apply disentanglement constraints (e.g., β-VAE, InfoGAN).
- Add axis alignment loss (match specific dimensions to interpretable features).
- Use linear probing or mutual information diagnostics on each axis.

---

## 111. Contrastive Loss Doesn’t Improve After Few Epochs

**Symptoms:**
- InfoNCE or triplet loss stagnates early.

**Suggested Next Steps:**
- Increase batch size or introduce cross-batch memory.
- Add hard negative mining or false positive suppression.
- Anneal temperature parameter to sharpen contrast.

---

## 112. Some Augmentations Hurt More Than Help

**Symptoms:**
- Certain transforms consistently degrade performance.

**Suggested Next Steps:**
- Log augmentation-wise performance (Ablation).
- Weight augmentations dynamically via learned augmentation policy (e.g., RandAugment, CTAugment).
- Use augmentation mixing or adaptive sampling.

---

## 113. Feature Collapse Occurs Only With Certain Optimizers

**Symptoms:**
- Model collapses only with, e.g., SGD but not Adam.

**Suggested Next Steps:**
- Tune momentum and weight decay separately per optimizer.
- Try Lookahead or LION optimizers for stability.
- Log parameter norm vs. gradient norm for each optimizer.

---

## 114. Embedding Space Is Dominated By Low-Level Statistics

**Symptoms:**
- PCA or t-SNE clusters reflect color, brightness, or length — not semantics.

**Suggested Next Steps:**
- Add style-invariant losses or cross-style contrastive training.
- Use domain adversarial training to remove low-level factors.
- Balance dataset by stratified sampling across confounds.

---

## 115. Representations Shift Over Epochs Without Improving Performance

**Symptoms:**
- Embeddings change (tracked via cosine sim, PCA, etc.), but validation accuracy or loss doesn’t improve.

**Suggested Next Steps:**
- Apply stability-promoting regularization (e.g., Temporal Consistency Loss).
- Add EMA smoothing to stabilize representations.
- Visualize class-wise or token-wise drift to localize the instability.

---

## 116. Multiple Augmented Views Collapse to the Same Point

**Symptoms:**
- Despite different views, embeddings end up nearly identical.

**Suggested Next Steps:**
- Increase augmentation strength, but diversify types (e.g., not just color + crop).
- Add view-specific noise or stochastic depth.
- Ensure contrastive or predictive loss compares only positive pairs, not self-matching.

---

## 117. Representations Encode Unintended Metadata (e.g., filename, sequence ID)

**Symptoms:**
- Clusters emerge based on metadata, not content.

**Suggested Next Steps:**
- Strip or randomize metadata at load time.
- Add confounder adversarial head to remove leaked info.
- Use information bottleneck or dropout on metadata paths.

---

## 118. Representation Collapse Only Happens With Gradient Accumulation

**Symptoms:**
- Training without accumulation is fine, but collapse appears when gradients are accumulated.

**Suggested Next Steps:**
- Reduce accumulation steps or scale learning rate per accumulation cycle.
- Normalize loss per micro-batch, not macro-batch.
- Track loss per step, not per epoch, to diagnose mismatch.

---

## 119. Loss Decreases on Synthetic Data but Not on Real Data

**Symptoms:**
- Strong learning signals on generated or proxy data, no generalization to real-world data.

**Suggested Next Steps:**
- Use domain adaptation loss (e.g., CORAL, MMD, domain confusion).
- Blend synthetic and real with style transfer, cycle consistency, or shared encoder.
- Add real-vs-synthetic discrimination loss to control blending.

---

## 120. Final Embeddings Lack Cluster Structure

**Symptoms:**
- All points are evenly distributed in space, no class or view-based clustering.

**Suggested Next Steps:**
- Add clustering loss (e.g., SwAV, DeepCluster).
- Apply contrastive loss across hard positives and negatives.
- Normalize embeddings to a hypersphere, then cluster.

---


