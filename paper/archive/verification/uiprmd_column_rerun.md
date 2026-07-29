# UI-PRMD column: raw (published) vs FK (repaired)

| row | raw | fk | delta |
|---|---|---|---|
| const-coord fraction | 0.5733 | 0.12 | -0.4533 |
| naive: naive_auroc | 0.5380 | 0.5270 | -0.0110 |
| naive: naive_shared_joints | 0.5380 | 0.5270 | -0.0110 |
| naive: naive_zscored | 0.5423 | 0.6088 | +0.0665 |
| conditions: Scratch | 0.5240 | 0.5293 | +0.0053 |
| conditions: Contrastive LP | 0.5179 | 0.5204 | +0.0024 |
| conditions: Contrastive FT | 0.5139 | 0.5322 | +0.0183 |
| conditions: Masked LP | 0.5119 | 0.5273 | +0.0154 |
| conditions: Masked FT | 0.5138 | 0.5245 | +0.0107 |
| allcorpora: Scratch | 0.5170 | 0.5267 | +0.0097 |
| allcorpora: Contrastive LP | 0.5321 | 0.5068 | -0.0252 |
| allcorpora: Contrastive FT | 0.5129 | 0.5219 | +0.0091 |
| allcorpora: Masked LP | 0.5320 | 0.5207 | -0.0113 |
| allcorpora: Masked FT | 0.5201 | 0.5293 | +0.0092 |
| robustness: ST-GCN backbone | 0.5135 | 0.5170 | +0.0034 |
| robustness: Rel.-joint input | 0.5199 | 0.5301 | +0.0103 |
| robustness: Per-seq. z-score | 0.5345 | 0.5485 | +0.0140 |
