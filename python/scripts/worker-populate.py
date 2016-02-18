#!/usr/local/bin/python3

"""
Populate relations.
Examples:
	worker populate pipeline.pre.ExtractSpikes
"""
from importlib import import_module
import sys
from argparse import ArgumentParser
import time
import numpy as np


def main(argv):
    parser = ArgumentParser(argv[0], description=__doc__)
    parser.add_argument('relation', type=str, help="Full import name of the relation.")
    parser.add_argument('--daemon', '-d', type=bool, help="Run in daemon mode, repeatedly checking.", default=False)
    parser.add_argument('--t_min', type=int, help="Minimal waiting time for daemon in sec.", default=5*60)
    parser.add_argument('--t_max', type=int, help="Maximal waiting time for daemon in sec.", default=15*60)

    args = parser.parse_args(argv[1:])

    p, m = args.relation.rsplit('.', 1)
    try:
        mod = import_module(p)
    except ImportError:
        print("Could not find module", p)
        return 1

    try:
        rel = getattr(mod, m)
    except AttributeError:
        print("Could not find class", m)
        return 1

    run_daemon = True # execute at least one loop
    while run_daemon:
        todo, done = rel().progress()
        if todo > 0:
            rel().populate(reserve_jobs=True)
        elif run_daemon:
            time.sleep(np.random.randint(args.t_min, args.t_max))
        run_daemon = args.daemon
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
