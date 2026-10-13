"""Minimal GR-HSTU GPU smoke: tiny HSTU + sampled-softmax, synthetic data.

Proves IAM + image pull + logging + CUDA + fbgemm. NOT a paper reproduction:
a few optimizer steps on random ids, then SMOKE OK. Reads hyperparameters
from CLI flags (SageMaker passes them as --key value).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fbgemm_gpu  # noqa: F401  (registers torch.ops.fbgemm)
import torch

from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    LocalNegativesSampler,
)
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    hstu_encoder,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)

D = 16
MAX_SEQ = 32
B = 4
NUM_ITEMS = 64


def build(device: torch.device):
    torch.manual_seed(1)
    emb = LocalEmbeddingModule(num_items=NUM_ITEMS, item_embedding_dim=D)
    sim, _ = get_similarity_function(
        module_type="DotProduct", query_embedding_dim=D, item_embedding_dim=D
    )
    model = hstu_encoder(
        max_sequence_length=MAX_SEQ,
        max_output_length=2,
        embedding_module=emb,
        similarity_module=sim,
        input_preproc_module=LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=MAX_SEQ + 2, embedding_dim=D, dropout_rate=0.0
        ),
        output_postproc_module=L2NormEmbeddingPostprocessor(embedding_dim=D, eps=1e-6),
        activation_checkpoint=False,
        verbose=False,
        num_blocks=2,
        num_heads=2,
        dqk=8,
        dv=8,
    )
    sampler = LocalNegativesSampler(
        num_items=NUM_ITEMS,
        item_emb=model._embedding_module._item_emb,
        all_item_ids=list(range(1, NUM_ITEMS + 1)),
        l2_norm=True,
        l2_norm_eps=1e-6,
    )
    loss_fn = SampledSoftmaxLoss(
        num_to_sample=8, softmax_temperature=0.05, model=model
    )
    return model.to(device), sampler.to(device), loss_fn.to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()

    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    assert torch.cuda.is_available(), "no CUDA on a GPU smoke job"
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)

    model, sampler, loss_fn = build(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    past_lengths = torch.tensor([8, 12, 16, 10], device=device)
    past_ids = torch.zeros(B, MAX_SEQ + 2, dtype=torch.long, device=device)
    for b, L in enumerate(past_lengths.tolist()):
        past_ids[b, :L] = torch.randint(1, NUM_ITEMS + 1, (L,), device=device)

    model.train()
    for step in range(args.steps):
        opt.zero_grad()
        in_emb = model.get_item_embeddings(past_ids)
        seq_emb = model(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=in_emb,
            past_payloads={},
        )
        sampler._item_emb = model._embedding_module._item_emb
        mask = past_ids[:, 1:] != 0
        loss, _ = loss_fn(
            lengths=past_lengths,
            output_embeddings=seq_emb[:, :-1, :],
            supervision_ids=past_ids[:, 1:],
            supervision_embeddings=in_emb[:, 1:, :],
            supervision_weights=mask.float(),
            negatives_sampler=sampler,
        )
        assert torch.isfinite(loss).all(), "non-finite loss"
        loss.backward()
        opt.step()
        print(f"[METRIC] step={step} loss={float(loss):.4f}", flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
