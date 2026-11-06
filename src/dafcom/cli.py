"""DAFCOM command-line interface; preparation never submits jobs."""

import argparse
import json
import logging

from . import workflow
from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("config")
        if name != "validate":
            command.add_argument("--stages", nargs="+", choices=list(workflow.DEPENDENCIES))
        if name in ("prepare", "run"):
            command.add_argument("--run-dir")
        if name == "run":
            command.add_argument("--workers", type=int, default=1)
    command = commands.add_parser("submit")
    command.add_argument("--run-dir", required=True)
    command = commands.add_parser("worker")
    command.add_argument("--run-dir", required=True)
    command.add_argument("--stage", required=True, choices=list(workflow.DEPENDENCIES))
    command.add_argument("--chunk", type=int, required=True)
    command.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "worker":
        workflow.worker(args.run_dir, args.stage, args.chunk, args.index)
    elif args.command == "submit":
        print(json.dumps(workflow.submit(args.run_dir), indent=2))
    else:
        config = load_config(args.config)
        if args.command == "validate":
            print("Configuration valid")
        elif args.command == "plan":
            print(
                json.dumps(
                    {
                        stage: len(tasks)
                        for stage, tasks in workflow.plan(config, args.stages).items()
                    },
                    indent=2,
                )
            )
        else:
            run_dir = workflow.prepare(config, args.run_dir, args.stages)
            print(run_dir)
            if args.command == "run":
                workflow.local_run(run_dir, args.workers)
    return 0
