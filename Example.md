## 9. Poor Latent Space Separation

**Short Version**

**Symptoms:**  
t-SNE or PCA shows all points clumped together.

**Suggestions:**  
- Improve augmentation diversity to avoid trivial alignment.  
- Add auxiliary tasks (e.g., rotation prediction or jigsaw).  
- Increase embedding dimensionality or training time.

---

**Longer Version**

**Problem:**  
Learned latent representations lack meaningful separation, causing different data points to be mapped close together in the feature space.

**Symptoms:**  
When visualizing the latent space using t-SNE or PCA, all data points appear tightly clustered, with little or no discernible separation between distinct classes or modes.

**Suggested Next Steps:**  
- **Increase the diversity of data augmentations.**  
  Broader and more varied augmentations encourage the model to learn robust, discriminative features rather than converging to trivial or redundant solutions.

- **Incorporate self-supervised auxiliary tasks.**  
  Tasks such as rotation prediction or jigsaw puzzles promote the learning of additional structure and help prevent feature collapse.

- **Expand the embedding dimensionality or extend training time.**  
  Increasing the capacity of the latent space or allowing more training iterations can provide the model with greater flexibility to separate different data distributions, provided sufficient data and regularization are available to avoid overfitting.

