import argparse
from dataclasses import replace

from mini_eval.registry import SUITES_DIR, fingerprint, load_registry, load_suite


def cmd_list(args):
    print("Models:")
    for m in load_registry().values():
        print(f"  {m.name:32s} {m.kind:7s} {m.family}")
    print("Suites:")
    for path in sorted(SUITES_DIR.glob("*.yaml")):
        print(f"  {path.stem}")


def cmd_fingerprint(args):
    model = load_registry()[args.model]
    if model.kind != "native":
        raise SystemExit(f"{model.name} is not a native model")
    values = fingerprint(model)
    print(f"sha256: {values['sha256']}")
    print(f"tokenizer_sha256: {values['tokenizer_sha256']}")


def cmd_run(args):
    from mini_eval.runner import run_suite

    registry = load_registry()
    names = list(registry) if args.all else args.models
    unknown = [n for n in names if n not in registry]
    if unknown or not names:
        raise SystemExit(f"Unknown or missing models: {unknown or 'none given'}. See `mini-eval list`.")
    suite = load_suite(args.suite)
    if args.limit is not None:
        suite = replace(suite, limit=args.limit or None)
    for name in names:
        print(f"== {name} on {suite.name} ==", flush=True)
        out = run_suite(registry[name], suite, args.device, args.allow_unpinned, args.require_clean)
        print(f"saved -> {out}")


def cmd_check(args):
    from mini_eval.checks import run_checks

    model = load_registry()[args.model]
    if model.kind != "native":
        raise SystemExit("Checks only apply to native models")
    raise SystemExit(0 if run_checks(model, args.device, args.allow_unpinned) else 1)


def cmd_report(args):
    from mini_eval.report import build_report

    print(f"saved -> {build_report(args.suite)}")


def main():
    ap = argparse.ArgumentParser(prog="mini-eval")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    p = sub.add_parser("fingerprint")
    p.add_argument("model")
    p.set_defaults(func=cmd_fingerprint)

    p = sub.add_parser("run")
    p.add_argument("suite")
    p.add_argument("models", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-unpinned", action="store_true")
    p.add_argument("--require-clean", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check")
    p.add_argument("model")
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-unpinned", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("report")
    p.add_argument("suite")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()