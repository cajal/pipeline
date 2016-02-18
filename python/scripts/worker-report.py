#!/usr/local/bin/python3

"""
Populate relations.
Examples:
	worker populate pipeline.pre.ExtractSpikes
"""
from argparse import ArgumentParser
from importlib import import_module
from slacker import Slacker
import time
import inspect
import sys
import pipeline
import datajoint as dj

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

def generate_report(module_name):
    module = import_module('pipeline.' + module_name)
    members = [m for m in dir(module) if not m.startswith('_')]
    if 'Jobs' in members:
        jobs = getattr(module, 'Jobs')
        members.remove('Jobs')
    for members in members:
        klass = getattr(module, member)
    # for member in inspect.getmembers(module):
    #     if isinstance(member, (dj.Computed, dj.Imported)):
    #         print(member)
    #     elif isinstance(member, dj.schema):
    #         schema = member


def main(argv):
    parser = ArgumentParser(argv[0], description=__doc__)
    parser.add_argument('--daemon', '-d', type=bool, help="Run in daemon mode, repeatedly checking.", default=False)
    parser.add_argument('--slack', type=str, help="Slack token to report successful population to slack.", default=None)
    parser.add_argument('--interval', type=int, help="Reporting interval in hours.", default=4)

    args = parser.parse_args(argv[1:])

    run_daemon = True # execute at least one loop

    while run_daemon:
        for pipeline_name, module_names in pipeline._report_on.items():
            for module_name in module_names:
                generate_report(module_name)
        run_daemon = args.daemon
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
