#!/usr/bin/env python3
from __future__ import annotations

import sys

from src.acquire import (
    download_exports,
    download_flywire_exports,
    download_flywire_mushroom_body_exports,
    require_raw_exports,
)
from src.config import (
    CONNECTOME_FLYWIRE_MUSHROOM_BODY,
    CONNECTOME_FLYWIRE_WHOLE,
    CONNECTOME_HEMIBRAIN_MUSHROOM_BODY,
    HEMIBRAIN_MB_ROI_LABELS,
    build_paths,
    parse_args,
)
from src.connectome import prepare_connectome
from src.plots import write_plots
from src.train import run_training
from src.validate import run_validation


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    paths = build_paths(cfg.output_dir, cfg.cache_dir)

    if cfg.mode in {"download", "all"}:
        try:
            require_raw_exports(paths)
            if cfg.mode == "all":
                print("Raw neuPrint exports already exist; reusing cached CSVs.")
            else:
                print("Raw neuPrint exports already exist; download step is complete.")
        except FileNotFoundError:
            if cfg.connectome == CONNECTOME_FLYWIRE_WHOLE:
                info = download_flywire_exports(
                    paths,
                    release=cfg.flywire_release,
                    download_dir=cfg.flywire_download_dir,
                )
            elif cfg.connectome == CONNECTOME_FLYWIRE_MUSHROOM_BODY:
                info = download_flywire_mushroom_body_exports(
                    paths,
                    release=cfg.flywire_release,
                    download_dir=cfg.flywire_download_dir,
                )
            elif cfg.connectome == CONNECTOME_HEMIBRAIN_MUSHROOM_BODY:
                info = download_exports(
                    paths,
                    requested_rois=HEMIBRAIN_MB_ROI_LABELS,
                    source_label=cfg.connectome,
                )
            else:
                info = download_exports(paths, source_label=cfg.connectome)
            print(
                f"Downloaded {cfg.connectome} exports: "
                f"{info['neuron_count']} neurons, {info['edge_count']} aggregated edges."
            )

    if cfg.mode in {"prepare", "all"}:
        graph = prepare_connectome(
            paths,
            signed_policy=cfg.signed_policy,
            connectome=cfg.connectome,
            whole_brain_pool_fraction=cfg.whole_brain_pool_fraction,
        )
        print(
            "Prepared graph: "
            f"N={graph.metadata['N']}, edges={graph.metadata['unsigned_edge_count']}, "
            f"primary={graph.metadata['primary_matrix']}, K={graph.metadata['estimated_K']}."
        )

    if cfg.mode in {"train", "all"}:
        metrics = run_training(paths, cfg.train, cfg.task)
        print(f"Training complete: wrote {len(metrics)} metric rows to {paths.metrics_by_seed_csv}.")

    if cfg.mode in {"validate", "all"}:
        run_validation(paths, cfg.task)
        write_plots(paths)
        print("Validation reports and plots are up to date.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
