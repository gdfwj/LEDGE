import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='LEDGE simulation')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Run a simulation')
    run.add_argument('config', type=Path)
    run.add_argument('--mode', choices=('fast', 'log'), default='fast')
    run.add_argument('--output', type=Path)
    run.add_argument('--requests-csv', type=Path)
    serve = commands.add_parser('serve', help='Start the GUI')
    serve.add_argument('--port', type=int, default=8765)
    serve.add_argument('--run-root', type=Path)
    args = parser.parse_args()
    if args.command == 'serve':
        from .server import serve
        serve(args.port, args.run_root)
        return
    from .simulation import run_simulation
    result = run_simulation(json.loads(args.config.read_text()), args.mode)
    if args.output:
        from .runs import write
        write(args.output, result)
    else:
        print(json.dumps(result['summary'], indent=2, ensure_ascii=False))
    if args.requests_csv:
        keys = ['request_id', 'instance', 'model', 'arrival_s', 'start_s', 'first_token_s', 'completion_s', 'queue_s', 'service_s', 'response_s', 'ttft_s', 'tpot_s']
        args.requests_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.requests_csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(result['requests'])


if __name__ == '__main__':
    main()
