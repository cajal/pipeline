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
    parser.add_argument('relations', type=str, help="List of full import names of relations that need to be populated (comma separated list).")
    parser.add_argument('--daemon', '-d', action='store_true', help="Run in daemon mode, repeatedly checking.")
    parser.add_argument('--t_min', type=int, help="Minimal waiting time for daemon in sec.", default=5*60)
    parser.add_argument('--t_max', type=int, help="Maximal waiting time for daemon in sec.", default=15*60)
    parser.add_argument('--restrictions','-r', type=str, help="Restrictions as a string.", default=None)

    args = parser.parse_args(argv[1:])

    rels_cls = {}
    for relname in map(lambda x: x.strip(), args.relations.split(',')):
        p, m = relname.rsplit('.', 1)
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

        rels_cls[relname] = rel

    run_daemon = True # execute at least one loop
    while run_daemon:
        for name, rel in rels_cls.items():
            if args.restrictions is not None:
                rel().populate(args.restrictions, reserve_jobs=True, suppress_errors=True)
            else:
                rel().populate(reserve_jobs=True, suppress_errors=True)

        run_daemon = args.daemon
        if run_daemon:
            t = np.random.randint(args.t_min, args.t_max)
            print('Going to sleeping for', t, 'seconds')
            time.sleep()

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
