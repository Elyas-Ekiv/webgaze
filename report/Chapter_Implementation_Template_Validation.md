# Template validation — Implementation & Results

Template: `c:\Users\Admin\Downloads\DS and AI-CSDS4105 - Final Review Report Template-Research (1).docx`
Output: `C:\Users\Admin\Desktop\course project\report\Final_Review_Report_with_Results.docx`

The requested Implementation & Results outline could not reuse template Chapter 4 numbering because that chapter is already **Preprocessing Datasets**. Content was added as **Chapter 7: IMPLEMENTATION & RESULTS** (sections 7.1–7.14) and checked against Chapters 1–6.

| Status | Check | Detail |
| --- | --- | --- |
| PASS | Chapter numbering | Template Chapter 4 is Preprocessing; Implementation & Results inserted as Chapter 7 before Appendices. |
| PASS | Datasets | Matches Ch5 implementation note: SALICON + UEyes + FiWi; WebSaliency excluded. |
| PASS | Image size | Uses 384×216 from Chapter 4 preprocessing. |
| PASS | Architecture | Dual-branch ResNet + structure + concat/attention fusion + U-Net decoder matches Chapter 5. |
| PASS | Training stages | SALICON pretrain → UEyes fine-tune → FiWi test matches Chapters 3 and 5. |
| PASS | Metrics | CC, NSS, AUC-Judd, SIM, KL-Divergence match Chapter 5 evaluation methods. |
| PASS | FiWi size | Uses held-out FiWi with 149 images, as stated in Chapter 3. |
| PASS | Best checkpoint | Saved best multimodal model is `concat` (CC 0.3146, AUC-Judd 0.7745). |
| PASS | Prototype | FastAPI upload/URL demo completes Chapter 6.2 optional web prototype. |
| NOTE | Chapter 6 Future Work | Chapter 6 still reads as a Sem-2 plan; Chapter 7 is the implemented outcome and can later be mirrored into an updated 6.1 conclusion. |
| NOTE | Dev environment wording | Chapter 3 mentions Jupyter/Colab; actual implementation used local Anaconda + Docker. Chapter 7 documents the environment used. |
| NOTE | Older WebSaliency sentences | Some early chapters still mention WebSaliency in plans; Chapter 5/7 correctly state it was not used. |

## Section map (requested → template report)

| Requested heading | Inserted as |
| --- | --- |
| 4.1 Implementation Overview | 7.1 |
| 4.2 Development Environment | 7.2 |
| 4.3 Dataset Implementation | 7.3 |
| 4.4 Model / Algorithm Implementation | 7.4 |
| 4.5 Model Training and Hyperparameter Configuration | 7.5 |
| 4.6 System / Application Implementation | 7.6 |
| 4.7 Experimental Setup | 7.7 |
| 4.8 Model Results | 7.8 |
| 4.9 Performance Evaluation | 7.9 |
| 3.10/4.10 Comparative Results | 7.10 |
| 3.11/4.11 Error / Failure Analysis | 7.11 |
| 3.12/4.12 System / Prototype Results | 7.12 |
| 3.13/4.13 Results Summary | 7.13 |
| 3.14/4.14 Chapter Summary | 7.14 |
