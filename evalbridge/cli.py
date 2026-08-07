"""
evalbridge CLI — report generation and version check.
"""

import argparse
import sys


def main():
    """Entry point for the evalbridge command."""
    parser = argparse.ArgumentParser(
        prog="evalbridge",
        description="evalbridge — A/B evaluation for ML models",
    )
    subparsers = parser.add_subparsers(dest="command")

    # evalbridge version
    subparsers.add_parser("version", help="Print the evalbridge version")

    # evalbridge report
    report_parser = subparsers.add_parser(
        "report", help="Generate an HTML report from a saved .evalbridge file"
    )
    report_parser.add_argument("path", help="Path to a .evalbridge experiment file")
    report_parser.add_argument(
        "--no-browser", action="store_true", help="Write report but do not open browser"
    )

    args = parser.parse_args()

    if args.command == "version":
        from evalbridge import __version__

        print(f"evalbridge {__version__}")

    elif args.command == "report":
        from evalbridge.experiment import Experiment

        try:
            exp = Experiment.load(args.path)
        except FileNotFoundError:
            print(f"Error: file not found: {args.path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading experiment: {e}", file=sys.stderr)
            sys.exit(1)

        result = exp.evaluate()
        open_browser = not args.no_browser
        result.report(path="report.html", open_browser=open_browser)
        print("Report saved to report.html")

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
