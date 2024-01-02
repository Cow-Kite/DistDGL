import argparse
import time

import dgl
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset

device = "cpu"

"""Load ogbn dataset."""
data = DglNodePropPredDataset(name="ogbn-products", root="./ogbn")
splitted_idx = data.get_idx_split()
g, labels = data[0]
labels = labels[:, 0]

g.ndata["features"] = g.ndata.pop("feat")
g.ndata["labels"] = labels
num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
#print(num_labels)

# Find the node IDs in the training, validation, and test set.
train_nid, val_nid, test_nid = (
    splitted_idx["train"],
    splitted_idx["valid"],
    splitted_idx["test"],
)

train_mask = th.zeros((g.num_nodes(),), dtype=th.bool)
train_mask[train_nid] = True
val_mask = th.zeros((g.num_nodes(),), dtype=th.bool)
val_mask[val_nid] = True
test_mask = th.zeros((g.num_nodes(),), dtype=th.bool)
test_mask[test_nid] = True
g.ndata["train_mask"] = train_mask
g.ndata["val_mask"] = val_mask
g.ndata["test_mask"] = test_mask
print(len(train_nid))
print(len(val_nid))
print(len(test_nid))

sampler = dgl.dataloading.NeighborSampler(
        [10,25]
    )

dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=1000,
        shuffle=True,
        drop_last=False,
    )

num_seeds = 0
num_inputs = 0
num = 0

# single training
# train_node: 196615
# batch_size: 1000
# step: train_node / batch_size (올림)

for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
    num += 1
    # Slice feature and label.
    batch_inputs = g.ndata["features"][input_nodes]
    batch_labels = g.ndata["labels"][seeds].long()
    num_seeds += len(blocks[-1].dstdata[dgl.NID])
    num_inputs += len(blocks[0].srcdata[dgl.NID])
    print(batch_inputs.size())
    # Move to target device.
    blocks = [block.to(device) for block in blocks]
    #print(blocks)
    batch_inputs = batch_inputs.to(device)
    batch_labels = batch_labels.to(device)

