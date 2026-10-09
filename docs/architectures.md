# Architecture reference

All aggregation methods use [`WindowClassifier`](../gnn/model.py), configured by `ModelConfig`. The architecture identifier is distinct from a trained model ID: the latter also identifies its experiment and run.

| Identifier | Implementation | Aggregation |
|---|---|---|
| `mean` | [readout.py](../gnn/readout.py) | Mean of raw node embeddings |
| `linear` | [linear_encoder.py](../gnn/linear_encoder.py) | Per-node linear projection, then mean |
| `pointwise_mlp` | [pointwise_mlp.py](../gnn/pointwise_mlp.py) | Per-node MLP, then mean |
| `mpnn` | [encoder.py](../gnn/encoder.py) | GraphSAGE over skeleton edges, then mean |
| `mpnn_complete` | [encoder.py](../gnn/encoder.py) | GraphSAGE over a complete window graph, then mean |
| `graph_transformer` | [graph_transformer.py](../gnn/graph_transformer.py) | Graph attention with CLS readout |
| `graph_transformer_teasar` | [graph_transformer_teasar.py](../gnn/graph_transformer_teasar.py) | Mixed embedded and structural skeleton nodes with CLS readout |

Classification uses an [LCPN hierarchical head](../gnn/lcpn.py) or a [flat head](../gnn/flat.py). The [ResNet trunk](../gnn/resnet.py) and positional, adjacency, embedding, and thickness options are recorded in the run configuration. Consult the implementation and `scripts/train_gnn.py --help` for supported combinations.

With the hierarchical head, the model forward pass returns a readout embedding; the head's `predict_top_down` method produces predictions. Existing dataset-specific inference entry points are listed in the [reproduction guide](reproducibility.md).
