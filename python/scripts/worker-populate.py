#!/usr/local/bin/python3

"""
Populate relations.
Examples:
	worker populate pipeline.pre.ExtractSpikes
"""
from importlib import import_module
import sys
from argparse import ArgumentParser
from slacker import Slacker
import time
import numpy as np


def post_message(msg, token=None, channel='#bot_planet'):
    """
    Posts a message to the specified slack channel if toke is not None
    :param msg: Message
    :param token: Slack integration token
    :param channel: slack channel (needs # in front)
    """
    if token is not None:
        slack = Slacker(token)
        slack.chat.post_message(channel=channel, text=msg, as_user=True)


def main(argv):
    parser = ArgumentParser(argv[0], description=__doc__)
    parser.add_argument('relation', type=str, help="Full import name of the relation.")
    parser.add_argument('--daemon', '-d', type=bool, help="Run in daemon mode, repeatedly checking.", default=False)
    parser.add_argument('--slack', type=str, help="Slack token to report successful population to slack.", default=None)
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
    if args.daemon:
        post_message("Starting population daemon for {relation}".format(relation=args.relation), token=args.slack)

    while run_daemon:
        keys = (rel().populated_from - rel()).project().fetch.as_dict()
        if len(keys) > 0:
            post_message("Found {no} unpopulated keys in {relation}. Starting to populate.".format(no=len(keys),
                                                                                                   relation=args.relation),
                         token=args.slack)
            rel().populate(reserve_jobs=True)
            msg = 'Just populated those keys in {relation}\n'.format(relation=args.relation) + \
                  '\n'.join(['{' + ', '.join([str(k) + ':' + str(v) for k, v in key.items()]) + '}' for key in keys])
            post_message(msg, token=args.slack)
        run_daemon = args.daemon
        if run_daemon: time.sleep(np.random.randint(args.t_min, args.t_max))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
