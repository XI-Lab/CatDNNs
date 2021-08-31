import numpy as np
import torch
import torch.nn as nn
atomref = np.zeros([36, 4])  # largest atomistic number is 35, U0 U H,G need atomref
atomref[6, 0] = -38.08212814
atomref[1, 0] = -0.601967955
atomref[8, 0] = -75.16077019
atomref[17, 0] = -460.1589048
atomref[35, 0] = -2571.561142
atomref[9, 0] = -99.80889669
atomref[5, 0] = -79.66130988
atomref[7, 0] = -0  # since N always show with B together

atomref[6, 1] = -37.93934655
atomref[1, 1] = -0.515270609
atomref[8, 1] = -74.83047132
atomref[17, 1] = -460.0680846
atomref[35, 1] = -2571.467721
atomref[9, 1] = -99.72059438
atomref[5, 1] = -79.42408612
atomref[7, 1] = -0  # since N always show with B together

atomref[6, 2] = -38.08162248
atomref[1, 2] = -0.601597709
atomref[8, 2] = -75.15870704
atomref[17, 2] = -460.1574369
atomref[35, 2] = -2571.559389
atomref[9, 2] = -99.80787199
atomref[5, 2] = -79.66002671
atomref[7, 2] = -0  # since N always show with B together

atomref[6, 3] = -38.08661108
atomref[1, 3] = -0.601088457
atomref[8, 3] = -75.16388831
atomref[17, 3] = -460.1589339
atomref[35, 3] = -2571.562024
atomref[9, 3] = -99.808346
atomref[5, 3] = -79.67131417
atomref[7, 3] = -0  # since N always show with B together

atomref = nn.Embedding.from_pretrained(torch.as_tensor(atomref), freeze=True)