"""t0 adapter (The Forecasting Company, `tfc-t0` package, T0Forecaster.predict)."""

from __future__ import annotations

import numpy as np
import torch

from tsbench.models.base import BaseAdapter, Forecast, ForecastTask

# predict() adjustments; all four on = the publisher's published-score setting (tfc-t0 README).
ADJUSTMENTS = ("trim_leading_missing", "symmetric_averaging", "clip_nonnegative", "snap_integers")


class T0Adapter(BaseAdapter):
    def load(self, params: dict, device: str) -> None:
        super().load(params, device)
        from t0 import T0Forecaster

        self.model = T0Forecaster.from_pretrained(self.card.source["hf_id"], revision=self.card.revision).to(device).eval()
        self.trained_levels = [round(float(q), 4) for q in self.model.config.quantile_levels]
        # ask for exactly the trained levels, so nothing is interpolated or extrapolated
        self.levels = self.trained_levels
        self.batch_size = int(self.params.get("batch_size", 16))
        self.adjustments = {k: bool(self.params.get(k, True)) for k in ADJUSTMENTS}

    def predict(self, batch: list[ForecastTask]) -> list[Forecast]:
        max_ctx = self.card.max_context
        out: list[Forecast | None] = [None] * len(batch)
        # One call needs equal context length and horizon: group.
        groups: dict[tuple[int, int], list[int]] = {}
        for i, t in enumerate(batch):
            n = len(t.context) if max_ctx is None else min(len(t.context), max_ctx)
            groups.setdefault((n, t.horizon), []).append(i)
        for (n, h), ids in groups.items():
            for s in range(0, len(ids), self.batch_size):
                sub = ids[s : s + self.batch_size]
                ctx = np.stack([batch[i].context.to_numpy(dtype=np.float32)[-n:] for i in sub])  # (B, T)
                future = None
                if batch[sub[0]].past_covariates is not None:
                    # All covariates are known-future: [B, F, T + H], numeric.
                    future = np.stack(
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
                with torch.inference_mode():
                    fc = self.model.predict(
                        ctx, horizon=h, quantile_levels=self.levels, future_covariates=future, **self.adjustments
                    )
                q = np.asarray(fc.quantiles.float().cpu().numpy() if hasattr(fc.quantiles, "cpu") else fc.quantiles)
                assert q.shape == (len(sub), h, len(self.levels)), q.shape  # (B, H, Q)
                for j, i in enumerate(sub):
                    out[i] = Forecast(
                        index=batch[i].horizon_index,
                        quantiles={lv: q[j, :, k].astype(np.float64) for k, lv in enumerate(self.levels)},
                        effective_context=int(ctx.shape[-1]),
                        context_finite=int(np.isfinite(ctx[j]).sum()),
                    )
        return out  # type: ignore[return-value]

    def runtime_info(self) -> dict:
        from importlib import metadata

        return {
            "tfc_t0": metadata.version("tfc-t0"),
            "quantile_levels": self.levels,
            "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "max_horizon_per_pass": int(self.model.max_horizon),
            **self.adjustments,
        }
