"""Shared Timestral adapter. The variant is selected ONLY by the card (hf_id + revision)."""

from __future__ import annotations

import numpy as np
import torch

from tsbench.models.base import BaseAdapter, Forecast, ForecastTask


class TimestralAdapter(BaseAdapter):
    def load(self, params: dict, device: str) -> None:
        super().load(params, device)
        from timestral import TimestralForecaster

        self.fc = TimestralForecaster.from_pretrained(
            self.card.source["hf_id"],
            device=device,
            revision=self.card.revision,
            max_context=self.card.max_context,
        )
        self.levels = [round(float(q), 4) for q in self.fc.quantile_levels]
        self.batch_size = int(self.params.get("batch_size", 16))
        self.symmetric_averaging = bool(self.params.get("symmetric_averaging", True))
        self.clip_nonnegative = bool(self.params.get("clip_nonnegative", True))

    def predict(self, batch: list[ForecastTask]) -> list[Forecast]:
        max_ctx = self.card.max_context
        out: list[Forecast | None] = [None] * len(batch)
        # Timestral batches need equal context length and horizon: group.
        groups: dict[tuple[int, int], list[int]] = {}
        for i, t in enumerate(batch):
            groups.setdefault((min(len(t.context), max_ctx), t.horizon), []).append(i)
        for (n, h), ids in groups.items():
            for s in range(0, len(ids), self.batch_size):
                sub = ids[s : s + self.batch_size]
                ctx = np.stack([batch[i].context.to_numpy(dtype=np.float32)[-n:] for i in sub])
                known = None
                if batch[sub[0]].past_covariates is not None:
                    # All covariates are known-future: one (B, C, T+H) array, NUMERIC (no categorical path).
                    known = np.stack(
                        [
                            np.concatenate(
                                [
                                    batch[i].past_covariates.to_numpy(dtype=np.float32)[-n:],
                                    batch[i].future_covariates.to_numpy(dtype=np.float32),
                                ]
                            ).T
                            for i in sub
                        ]
                    )
                fc = self.fc.forecast(
                    ctx,
                    prediction_length=h,
                    known_covariates=known,
                    symmetric_averaging=self.symmetric_averaging,
                    clip_nonnegative=self.clip_nonnegative,
                )
                q = fc.quantiles.float().cpu().numpy()  # (B, 1, H, Q)
                assert q.shape == (len(sub), 1, h, len(self.levels)), q.shape
                for j, i in enumerate(sub):
                    out[i] = Forecast(
                        index=batch[i].horizon_index,
                        quantiles={lv: q[j, 0, :, k].astype(np.float64) for k, lv in enumerate(self.levels)},
                        effective_context=int(ctx.shape[-1]),
                        context_finite=int(np.isfinite(ctx[j]).sum()),
                    )
        return out  # type: ignore[return-value]

    def runtime_info(self) -> dict:
        return {
            "timestral_max_context": self.fc.max_context,
            "symmetric_averaging": self.symmetric_averaging,
            "clip_nonnegative": self.clip_nonnegative,
            "quantile_levels": self.levels,
            "torch_threads": torch.get_num_threads(),
        }
