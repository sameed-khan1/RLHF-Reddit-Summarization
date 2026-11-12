
import torch


def _pad(seqs, pad_id, side):
    length = max(len(s) for s in seqs)

    ids = torch.full(
        (len(seqs), length),
        pad_id,
        dtype=torch.long
    )

    mask = torch.zeros(
        (len(seqs), length),
        dtype=torch.long
    )

    for i, seq in enumerate(seqs):
        tensor = torch.tensor(seq, dtype=torch.long)

        if side == "right":
            ids[i, :len(seq)] = tensor
            mask[i, :len(seq)] = 1
        else:
            ids[i, length - len(seq):] = tensor
            mask[i, length - len(seq):] = 1

    return ids, mask


def collate_sft(batch, pad_id):
    seqs = [
        item["prompt_ids"] + item["summary_ids"]
        for item in batch
    ]

    ids, mask = _pad(seqs, pad_id, "right")

    prompt_lens = torch.tensor([
        len(item["prompt_ids"])
        for item in batch
    ])

    return {
        "input_ids": ids,
        "attention_mask": mask,
        "prompt_lens": prompt_lens
    }


def collate_pref(batch, pad_id):
    chosen, chosen_mask = _pad(
        [
            item["prompt_ids"] + item["chosen_ids"]
            for item in batch
        ],
        pad_id,
        "right"
    )

    rejected, rejected_mask = _pad(
        [
            item["prompt_ids"] + item["rejected_ids"]
            for item in batch
        ],
        pad_id,
        "right"
    )

    prompt_lens = torch.tensor([
        len(item["prompt_ids"])
        for item in batch
    ])

    return {
        "chosen_ids": chosen,
        "chosen_mask": chosen_mask,
        "rejected_ids": rejected,
        "rejected_mask": rejected_mask,
        "prompt_lens": prompt_lens
    }


def collate_prompts(batch, pad_id):
    ids, mask = _pad(
        [item["prompt_ids"] for item in batch],
        pad_id,
        "left"
    )

    return {
        "input_ids": ids,
        "attention_mask": mask
    }
