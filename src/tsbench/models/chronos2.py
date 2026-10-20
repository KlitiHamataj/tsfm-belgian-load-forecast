"""Chronos-2 adapter (official `chronos-forecasting` package, Chronos2Pipeline.predict)."""

from __future__ import annotations

import numpy as np
import torch

from tsbench.models.base import BaseAdapter, Forecast, ForecastTask


class Chronos2Adapter(BaseAdapter):
    def load(self, params: dict, device: str) -> None:
        super().load(params, device)
        from chronos import Chronos2Pipeline

        self.pipe = Chronos2Pipeline.from_pretrained(
            self.card.source["hf_id"], device_map=device, revision=self.card.revision
        )
        self.levels = [round(float(q), 4) for q in self.pipe.quantiles]
        self.batch_size = int(self.params.get("batch_size", 64))

    def predict(self, batch: list[ForecastTask]) -> list[Forecast]:
        max_ctx = self.card.max_context
        inputs, fed = [], []
        for t in batch:
            n = min(len(t.context), max_ctx)  # truncate ourselves so "effective" is what we feed
            target = t.context.to_numpy(dtype=np.float32)[-n:]
            item: dict = {"target": target}
            if t.past_covariates is not None:
                assert t.future_covariates is not None
                item["past_covariates"] = {
                    c: t.past_covariates[c].to_numpy(dtype=np.float32)[-n:] for c in t.past_covariates
                }
                item["future_covariates"] = {
                    c: t.future_covariates[c].to_numpy(dtype=np.float32) for c in t.future_covariates
                }
            inputs.append(item)
            fed.append((len(target), int(np.isfinite(target).sum())))
        # Tasks in a chunk share horizon except on DST days; group by horizon.
        out: list[Forecast | None] = [None] * len(batch)
        for h in sorted({t.horizon for t in batch}):
            ids = [i for i, t in enumerate(batch) if t.horizon == h]
            with torch.inference_mode():
                preds = self.pipe.predict(
                    [inputs[i] for i in ids],
                    prediction_length=h,
                    batch_size=self.batch_size,
                    context_length=max_ctx,
                )
            for i, p in zip(ids, preds):
                arr = p.float().numpy()  # (n_variates=1, n_quantiles, h)
                assert arr.shape == (1, len(self.levels), h), arr.shape
                out[i] = Forecast(
                    index=batch[i].horizon_index,
                    quantiles={q: arr[0, k].astype(np.float64) for k, q in enumerate(self.levels)},
                    effective_context=fed[i][0],
                    context_finite=fed[i][1],
                )
        return out  # type: ignore[return-value]

    def runtime_info(self) -> dict:
        import chronos

        return {"chronos_forecasting": getattr(chronos, "__version__", "n/a"), "quantile_levels": self.levels}
