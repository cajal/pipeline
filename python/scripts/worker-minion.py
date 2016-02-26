#!/usr/local/bin/python3

"""
Populate relations.
Examples:
	worker populate pipeline.pre.ExtractSpikes
"""
import pickle
import sys
from argparse import ArgumentParser
from importlib import import_module
from datajoint.relational_operand import AndList
from pipeline.minions import SIMPLE_POPULATOR, Gru


def callback(ch, method, properties, body):
    order = pickle.loads(body)
    print("\tReceived order to populate {relation}".format(relation=order['relation']))
    relation = order['relation']
    p, m = relation.rsplit('.', 1)
    try:
        mod = import_module(p)
    except ImportError:
        print("Could not find module", p)
        return 1

    try:
        rel = getattr(mod, m)()
    except AttributeError:
        print("Could not find class", m)
        return 1

    restrictions = order['restrictions']
    a = AndList(rel.heading)
    for r in restrictions:
        a.add(r)
    rel.populate(reserve_jobs=True, restriction = a)
    print("Executed populateion of ", relation, ' with restrictions ', a)
    ch.basic_ack(delivery_tag = method.delivery_tag)



def main(argv):
    parser = ArgumentParser(argv[0], description=__doc__)
    parser.add_argument('host', type=str, help="Hostname of rabbitmq.")
    parser.add_argument('--queue', type=str, help="Queue at rabbitmq.", default=SIMPLE_POPULATOR)

    args = parser.parse_args(argv[1:])

    master = Gru(host=args.host, queue=args.queue)

    while True:
        try:
            with master.connection() as channel:
                print(' Minion is eagerly awaiting commands. To exit press CTRL+C to kill that minion.')
                channel.basic_qos(prefetch_count=1)
                channel.basic_consume(callback, queue=args.queue, no_ack=False)
                channel.start_consuming()
        except:
            raise

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
