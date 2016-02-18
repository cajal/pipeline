#!/usr/local/bin/python3

"""
Populate relations.
Examples:
	worker populate pipeline.pre.ExtractSpikes
"""
import os
import random
from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module
from slacker import Slacker
import time
import inspect
import sys
import pipeline
import datajoint as dj

addr = ['Despicable masters', 'Evil overlords', 'Dear Villain Scientists', 'Ladies and Gentlemen',
        'Fellow Slackers', '"Real" people', 'Dear Biomass', 'Datajoyous greetings']


def post_report(comment, report, token=None, channel='#bot_planet'):
    if token is not None:
        slack = Slacker(token)
        with open('report.txt', 'w') as fid:
            fid.write(report)
        slack.files.upload('report.txt', filetype='text', title='Minion report', initial_comment=comment,
                           channels=channel)
        # os.remove('./report.txt')


def progress_bar(k, n, barlen=20):
    p = 1 if n == 0 else (n - k) / n
    eq = '=' * int(round(p * barlen))
    spaces = ' ' * (barlen - len(eq))
    return "[%s] %.1f%% (%i/%i)" % (eq + spaces, p * 100, n - k, n)


def running_jobs(schema, module_name, rel_name):
    running = schema.jobs & dict(table_name=module_name + '.' + rel_name)
    error = len(running & 'status="error"')
    running = len(running) - error
    ret = "Minions: {0} working".format(running) + ", {0} in error".format(error) * (error > 0)
    if len(ret) > 0:
        ret = '(' + ret + ')'
    return ret


def generate_report(module_name):
    module = import_module('pipeline.' + module_name)

    schema = inspect.getmembers(module, lambda x: isinstance(x, dj.schema))[0][1]

    ret = []
    for name, klass in inspect.getmembers(module,
                                          lambda x: isinstance(x, type) and issubclass(x, (dj.Computed, dj.Imported))):
        prog = klass().progress(display=False)

        if prog[0] > 0:
            line = '{:<28}  {:<43}  {:<30}'.format(module_name + '.' + name + ': ',
                                                   progress_bar(*prog), running_jobs(schema, module_name, name))
            ret.append(line)
    return '\n'.join(ret)


def main(argv):
    parser = ArgumentParser(argv[0], description=__doc__)
    parser.add_argument('--daemon', '-d', type=bool, help="Run in daemon mode, repeatedly checking.", default=False)
    parser.add_argument('--slack', type=str, help="Slack token to report successful population to slack.", default=None)
    parser.add_argument('--interval', type=int, help="Reporting interval in hours.", default=4)

    args = parser.parse_args(argv[1:])

    run_daemon = True  # execute at least one loop

    while run_daemon:
        address = "{0}, here is the report from ".format(random.choice(addr)) + "*" + datetime.now().strftime(
            "%A, %d. %B %Y %I:%M%p") + "*"
        report = []
        for pipeline_name, module_names in pipeline._report_on.items():
            report.append("_{0}_:".format(pipeline_name))
            report.extend([generate_report(module_name) for module_name in module_names])
            report.append('\n')

        report = '\n'.join(report)
        post_report(address, report, token=args.slack)
        run_daemon = args.daemon
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
