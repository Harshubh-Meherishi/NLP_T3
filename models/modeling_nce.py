from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InfoNCEOutput:
    logits: torch.Tensor
    selected: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hits_at_1: Optional[torch.Tensor] = None
    easy_loss: Optional[torch.Tensor] = None

def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

def concat_padded_tensors(t1: torch.Tensor, t2: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    assert t1.shape[:-1] == t2.shape[:-1], "The shapes of t1 and t2 must match except for the last dimension."
    concat_dim = t1.shape[-1] + t2.shape[-1]

    # Reshape t1 and t2
    v1 = t1.reshape(-1, t1.shape[-1])
    v2 = t2.reshape(-1, t2.shape[-1])

    # Concatenate t1 and t2
    out = torch.cat([v1, v2], dim=-1)

    # Create a tensor of the same shape as out filled with pad_token
    pad_tensor = v1.new_full((v1.shape[0], concat_dim), pad_token)

    # Replace the values in pad_tensor with the values from out where they are not equal to pad_token
    pad_tensor_mask = (out != pad_token)
    pad_tensor[pad_tensor_mask] = out[pad_tensor_mask]

    # Reshape the final tensor and return it
    return pad_tensor.reshape(*t1.shape[:-1], concat_dim)

def pad_tensor(tensor_to_pad: torch.Tensor, new_size: int = 0, pad_token: int = 0) -> torch.Tensor:
    if tensor_to_pad.shape[-1] >= new_size:
        return tensor_to_pad

    pad_shape = tensor_to_pad.shape[:-1] + (new_size - tensor_to_pad.shape[-1],)
    pad = tensor_to_pad.new_full(pad_shape, pad_token)
    return torch.cat([tensor_to_pad, pad], dim=-1)

def tanh_clip(x, clip_val=None):
    if clip_val is not None:
        return torch.clamp(x, -clip_val, clip_val)
    else:
        return x


def calc_nce_regularizer(scores, regularizer_coef=4e-2):
    return regularizer_coef * torch.mean(scores ** 2.0)

class InfoNCE(nn.Module):
    def __init__(
        self,
        model,
        pad_token_id: int,
        inbatch_negatives: bool = False,
        demi: bool = False,
        encoder_emb_method: str = "first_token",
        clip_val: float = 100.0,
        project: Optional[str] = None,
    ):
        super().__init__()
        
        self.model = model
        self.pad_token_id = pad_token_id
        self.inbatch_negatives = inbatch_negatives
        self.demi = demi
        self.encoder_emb_method = encoder_emb_method
        
        self.clip_val = clip_val
        self.mlp = None
        
        if project is not None:
            if project == "linear":
                self.mlp = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
                    nn.ReLU()
                )

    def forward(
        self,
        history_input_ids,
        positive_input_ids,
        negative_input_ids,
        history_token_type_ids=None,
        positive_token_type_ids=None,
        negative_token_type_ids=None,
        history_attention_mask=None,
        positive_attention_mask=None,
        negative_attention_masks=None,
    ):
        assert history_input_ids.shape[0] == positive_input_ids.shape[0], "history_ids and positive_ids must share the first dim"
        assert history_input_ids.shape[0] == negative_input_ids.shape[0], "history_ids and negative_ids must share the first dim"

        B = history_input_ids.shape[0]

        if self.model.config.is_encoder_decoder:
            candidates, history_hidden_states = self._get_queries_and_candidates_encoder_decoder(
                history_input_ids,
                history_attention_mask,
                positive_input_ids,
                positive_attention_mask,
                negative_input_ids,
                negative_attention_masks,
            )
        else:
            candidates, history_hidden_states = self._get_queries_and_candidates_decoder_only(
                history_input_ids,
                history_attention_mask,
                history_token_type_ids,
                positive_input_ids,
                positive_attention_mask,
                positive_token_type_ids,
                negative_input_ids,
                negative_attention_masks,
                negative_token_type_ids,
            )

        negative_mask = (negative_input_ids != self.pad_token_id).sum(-1).sum(-1) > 0
        H = history_hidden_states.shape[-1]

        scores = torch.bmm(candidates, history_hidden_states.unsqueeze(1).transpose(1, 2)).squeeze(-1) / np.sqrt(H)

        easy_nce_loss = None
        if self.demi or self.inbatch_negatives:
            inbatch_mask = (1 - torch.eye(B, device=scores.device)).unsqueeze(-1).expand(B, B, candidates.shape[1]).reshape(B, -1)
            mask = torch.cat([torch.ones_like(scores), inbatch_mask], dim=-1)

            inbatch_scores = torch.mm(history_hidden_states, candidates.view(-1, candidates.shape[-1]).T) / np.sqrt(H)

            if self.demi:
                easy_scores = masked_log_softmax(inbatch_scores, inbatch_mask)
                if self.stabilize:
                    easy_scores = tanh_clip(easy_scores, self.clip_val)
                    easy_reg = calc_nce_regulaizer(easy_scores, self.regularizer_coef)
                else:
                    easy_reg = 0.0
                easy_nce_loss = -easy_scores[negative_mask, 0].mean() + easy_reg
                scores = F.log_softmax(scores, dim=-1)
            else:
                scores = masked_log_softmax(torch.cat([scores, inbatch_scores], dim=-1), mask)
        else:
            scores = F.log_softmax(scores, dim=-1)

        if self.stabilize:
            scores = tanh_clip(scores, self.clip_val)
            reg = calc_nce_regulaizer(scores, self.regularizer_coef)
        else:
            reg = 0.0

        _, max_score_indices = torch.max(scores, dim=1)
        selected_cands = max_score_indices

        hits_at_1 = (max_score_indices[negative_mask] == 0).float().sum() / negative_mask.int().sum()
        nce_loss = -scores[negative_mask, 0].mean() + reg

        return InfoNCEOutput(
            scores=scores,
            selected_cands=selected_cands,
            nce_loss=nce_loss,
            hits_at_1=hits_at_1,
            easy_nce_loss=easy_nce_loss,
        )

    def _get_queries_and_candidates_decoder_only(self, history_input_ids, history_attention_mask,
                                              history_token_type_ids, positive_input_ids, positive_attention_mask,
                                              positive_token_type_ids, negative_input_ids, negative_attention_masks,
                                              negative_token_type_ids):
        B, Lh = history_input_ids.shape

        # Get lengths of non-padded tokens in the input sequences
        history_lengths = torch.sum(history_input_ids != self.pad_token_id, dim=-1)
        positive_lengths = torch.sum(positive_input_ids != self.pad_token_id, dim=-1)
        negative_lengths = torch.sum(negative_input_ids != self.pad_token_id, dim=-1)

        # Get the hidden states of the last layer of the Transformer for each input sequence
        history_output = self.model(input_ids=history_input_ids, token_type_ids=history_token_type_ids,
                                    attention_mask=history_attention_mask, output_hidden_states=True)
        history_hidden_states = history_output.hidden_states[-1][torch.arange(B), history_lengths - 1]

        pos_output = self.model(input_ids=positive_input_ids, token_type_ids=positive_token_type_ids,
                                attention_mask=positive_attention_mask, output_hidden_states=True)
        pos_hidden_states = pos_output.hidden_states[-1][torch.arange(B), positive_lengths - 1]

        neg_output = self.model(input_ids=negative_input_ids, token_type_ids=negative_token_type_ids,
                                attention_mask=negative_attention_masks, output_hidden_states=True)
        neg_hidden_states = neg_output.hidden_states[-1].view(B, -1, self.model.config.hidden_size)[
            torch.arange(B), negative_lengths - 1]

        # Apply the MLP if it exists
        if self.mlp is not None:
            pos_hidden_states = self.mlp(pos_hidden_states)
            neg_hidden_states = self.mlp(neg_hidden_states)
            history_hidden_states = self.mlp(history_hidden_states)

        # Concatenate the positive and negative hidden states and return them along with the history hidden state
        candidates = torch.cat([pos_hidden_states.unsqueeze(1), neg_hidden_states], dim=1)
        return candidates, history_hidden_states

    def _get_queries_and_candidates_encoder_decoder(
        self,
        history_input_ids,
        history_attention_mask,
        positive_input_ids,
        positive_attention_mask,
        negative_input_ids,
        negative_attention_masks,
    ):
        # Get the shapes of the input tensors
        B, Lh = history_input_ids.shape
        N = negative_input_ids.shape[1]
        Lp = positive_input_ids.shape[-1]
        Ln = negative_input_ids.shape[-1]

        # Prepare the input tensors for the model
        history_output = self.model(
            input_ids=history_input_ids,
            attention_mask=history_attention_mask,
            decoder_input_ids=positive_input_ids[:, 0].unsqueeze(-1),
            decoder_attention_mask=positive_attention_mask[:, 0].unsqueeze(-1),
            output_hidden_states=True,
        )

        # Compute the history_hidden_states based on the specified encoder_emb_method
        if self.encoder_emb_method == "mean_pool":
            history_mask = (
                (history_input_ids != self.pad_token_id)
                .int()
                .unsqueeze(-1)
                .expand_as(history_output.encoder_last_hidden_state)
            )
            history_lengths = (history_input_ids != self.pad_token_id).sum(-1).unsqueeze(-1)
            history_hidden_states = (
                torch.sum(history_output.encoder_last_hidden_state * history_mask, dim=1) / history_lengths
            )
        elif self.encoder_emb_method == "dec_first":
            history_hidden_states = history_output.decoder_hidden_states[-1][:, 0]
        else:
            history_hidden_states = history_output.encoder_last_hidden_state[:, 0]

        # Pad the input tensors to have the same length
        if Ln > Lp:
            positive_input_ids = pad_tensor(positive_input_ids, negative_input_ids.shape[-1], self.pad_token_id)
            positive_attention_mask = pad_tensor(positive_attention_mask, negative_input_ids.shape[-1])
        elif Lp > Ln:
            negative_input_ids = pad_tensor(negative_input_ids, positive_input_ids.shape[-1], self.pad_token_id)
            negative_attention_masks = pad_tensor(negative_attention_masks, positive_input_ids.shape[-1])

        # Concatenate the positive and negative input tensors and prepare them for the model
        decoder_input_ids = torch.cat(
            [positive_input_ids.unsqueeze(1), negative_input_ids],
            dim=1,
        ).view(B * (N + 1), -1)
        decoder_attention_mask = torch.cat(
            [positive_attention_mask.unsqueeze(1), negative_attention_masks], dim=1
        ).view(B * (N + 1), -1)
        decoder_lengths = (decoder_input_ids != self.pad_token_id).sum(-1)

        # Run the model to get the candidates and apply the MLP if specified
        output = self.model(
            input_ids=history_input_ids.unsqueeze(1).expand(B, N + 1, Lh).reshape(-1, Lh),
            attention_mask=history_attention_mask.unsqueeze(1).expand(B, N + 1, Lh).reshape(-1, Lh),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        candidates = output.decoder_hidden_states[-1][torch.arange(B * (N + 1)), decoder_lengths - 1]
        if self.mlp is not None:
            candidates = self.mlp(candidates)
            history_hidden_states = self.mlp(history_hidden_states)

        return candidates.view(B, N + 1, -1), history_hidden_states
