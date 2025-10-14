import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

dataset = pl.datasets.WildGuardMixDataset(split="train")
train_dataset, val_dataset = dataset.split(0.8)
test_dataset = pl.datasets.WildGuardMixDataset(split="test")

print(f"Train dataset: {len(train_dataset)}; Test dataset: {len(test_dataset)}")

probes = {
    "logistic": pl.probes.Logistic(
        layer=40, sequence_pooling=pl.SequencePooling.MEAN, C=0.01
    ),
    "mlp": pl.probes.MLP(layer=40, sequence_pooling=pl.SequencePooling.MEAN),
}

pl.scripts.train_probes(
    probes=probes,
    dataset=train_dataset,
    model=model,
    tokenizer=tokenizer,
    batch_size=8,
    mask=pl.masks.user(),
)

predictions, metrics = pl.scripts.evaluate_probes(
    probes=probes,
    dataset=test_dataset,
    model=model,
    tokenizer=tokenizer,
    mask=pl.masks.user(),
    batch_size=8,
    metrics=[
        pl.metrics.f1,
        pl.metrics.auroc,
    ],
)

pl.visualization.print_metrics(metrics)
