#!/usr/bin/env python3
"""Launch MiloNet's Gradio UI against an isolated prepared corpus."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


RUNTIME_FILES = (
    "all_doc_tool.txt",
    "file_structure.txt",
    "global_id_map.json",
    "tool_access_info.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the existing MiloNet UI with a prepared test corpus."
    )
    parser.add_argument(
        "--structured-original",
        required=True,
        help="Prepared Original_documents-style folder.",
    )
    parser.add_argument(
        "--processed-documents",
        required=True,
        help="Prepared Processed_documents-style folder.",
    )
    parser.add_argument(
        "--runtime-dir",
        required=True,
        help="Isolated runtime containing registry files and Chroma stores.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Allow Gradio to create a public share link. Disabled by default.",
    )
    return parser.parse_args()


def validate_inputs(
    structured_original: Path,
    processed_documents: Path,
    runtime_dir: Path,
) -> None:
    for label, path in (
        ("structured original", structured_original),
        ("processed documents", processed_documents),
        ("runtime", runtime_dir),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} folder does not exist: {path}")

    processed_map = processed_documents / "processed_files_map.json"
    if not processed_map.is_file():
        raise FileNotFoundError(
            f"processed document registry does not exist: {processed_map}"
        )

    missing_runtime_files = [
        name for name in RUNTIME_FILES if not (runtime_dir / name).is_file()
    ]
    if missing_runtime_files:
        raise FileNotFoundError(
            "runtime is incomplete; missing: " + ", ".join(missing_runtime_files)
        )


def ensure_modes_link(runtime_dir: Path, project_dir: Path) -> None:
    source = next(
        (candidate for candidate in (project_dir / "Modes", project_dir / "modes") if candidate.is_dir()),
        None,
    )
    if source is None:
        raise FileNotFoundError(f"Mode definitions do not exist under: {project_dir}")
    target = runtime_dir / "Modes"
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"runtime Modes path points elsewhere: {target}")
        return
    target.symlink_to(source, target_is_directory=True)


def ensure_runtime_config(runtime_dir: Path, project_dir: Path) -> None:
    source = project_dir / "parameters.ini"
    target = runtime_dir / "parameters.ini"
    if target.exists():
        return
    if not source.is_file():
        raise FileNotFoundError(f"MiloNet configuration does not exist: {source}")
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    structured_original = Path(args.structured_original).expanduser().resolve()
    processed_documents = Path(args.processed_documents).expanduser().resolve()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()

    validate_inputs(structured_original, processed_documents, runtime_dir)
    ensure_modes_link(runtime_dir, project_dir)
    ensure_runtime_config(runtime_dir, project_dir)

    # helper.py binds its Chroma clients during import, so set this first.
    os.environ["MILONET_RUNTIME_DIR"] = str(runtime_dir)

    import gradio as gr
    import MiloNet

    MiloNet.original_documents_folder = structured_original
    MiloNet.processed_documents_folder = processed_documents

    original_launch = gr.Blocks.launch

    def launch_isolated(blocks, *launch_args, **launch_kwargs):
        launch_kwargs["share"] = args.share
        launch_kwargs["server_name"] = args.host
        launch_kwargs["server_port"] = args.port
        return original_launch(blocks, *launch_args, **launch_kwargs)

    gr.Blocks.launch = launch_isolated

    old_cwd = Path.cwd()
    try:
        os.chdir(runtime_dir)
        interface = MiloNet.build_gradio_interface()
        print(f"Starting isolated MiloNet UI at http://{args.host}:{args.port}")
        interface.run()
    finally:
        gr.Blocks.launch = original_launch
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
