# Experimental results

Best multimodal checkpoint: `concat`

FiWi images: 149

| Model | CC | NSS | AUC-Judd | SIM | KL-Divergence |
| --- | --- | --- | --- | --- | --- |
| center_bias | 0.1817 | 0.5952 | 0.6734 | 0.2422 | 2.1649 |
| resnet_only | 0.2793 | 0.9599 | 0.7606 | 0.2873 | 1.7724 |
| mlp_only | 0.1533 | 0.5011 | 0.6300 | 0.1994 | 1.9998 |
| transformer_only | 0.1568 | 0.5143 | 0.6372 | 0.2093 | 1.9704 |
| vit_only | 0.1652 | 0.5400 | 0.6540 | 0.2008 | 1.9928 |
| concat | 0.3146 | 1.0902 | 0.7745 | 0.2974 | 1.6580 |
| attention | 0.3037 | 1.0553 | 0.7598 | 0.2960 | 1.7608 |

Report: `C:\Users\Admin\Desktop\course project\report\Final_Review_Report_with_Results.docx`