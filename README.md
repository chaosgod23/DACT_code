# DACT_code
Data-centric Cross-city Knowledge Transfer for Regional Socioeconomic Prediction
## 核心摘要 (Abstract)

Urban region representation learning is essential for **regional socioeconomic prediction**, enabling intelligent applications like crime forecasting, GDP predicting, and urban planning. However, generalizing these representations to data-scarce cities remains a challenge due to severe **spatial semantic heterogeneity**. 

Existing embedding-centric approaches typically employ a decoupled strategy: they first learn city-specific representations independently and then attempt to align them via post-hoc mapping functions. This process often results in limited transferability, as the initial embeddings in data-scarce cities are inherently noisy and structurally incompatible.

---

### 🚀 UniUrban: A Data-Centric Framework

In response, we propose **UniUrban**, a novel *data-centric* framework that shifts the paradigm from **embedding alignment** to **structural data alignment** prior to learning. 

* **Semantic Bridge:** UniUrban constructs a shared spatial semantic graph by leveraging **Large Language Models (LLMs)** to resolve heterogeneity across *schema*, *domain*, and *entity* levels.
* **Pairwise Region Coordination:** We introduce a coordination learning mechanism to enable robust knowledge transfer. This strategy jointly captures intra-city specificity and inter-city generalities:
    $$\mathcal{L}_{coord} = \sum_{(r_i, r_j) \in \mathcal{P}} \phi(z_i, z_j; \theta)$$
    *(注：此处可根据你论文中的实际公式替换 $\mathcal{L}_{coord}$ 的定义)*

### 📈 实验表现 (Experimental Results)

Extensive experiments on **NYC, Chicago, and Shenzhen** across five prediction tasks demonstrate that **UniUrban** significantly outperforms state-of-the-art baselines. In low-resource settings, it achieves:

| Metric | Improvement (vs. SOTA) |
| :--- | :--- |
| **MAE Reduction** | $\downarrow 14.8\%$ |
| **RMSE Reduction** | $\downarrow 13.5\%$ |
| **$R^2$ Improvement** | $\uparrow 155.1\%$ |

> **Conclusion:** These results validate the superiority of the **data-centric paradigm** for robust regional socioeconomic prediction.
