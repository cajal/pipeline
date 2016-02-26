from contextlib import contextmanager
import pika
import pickle
from . import SIMPLE_POPULATOR
import inspect

class Gru:
    def __init__(self, host, queue=SIMPLE_POPULATOR):
        self.host = host
        self.queue = queue

    @contextmanager
    def connection(self):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=self.host, heartbeat_interval=0))
            channel = connection.channel()
            channel.queue_declare(queue=self.queue, durable=True)
            yield channel
        except:
            connection.close()
            raise
        finally:
            connection.close()

    def populate(self, relation_cls, *restrictions):
        rel = relation_cls()
        todo = rel.populated_from
        todo.restrict(*restrictions)
        task_list =  (todo - rel.target.project())
        relation = inspect.getmodule(relation_cls).__name__ + '.' + relation_cls.__name__
        with self.connection() as channel:
            task_counter = 0
            for task in task_list.fetch.keys():
                message = pickle.dumps({'relation': relation, 'restrictions': restrictions + (task,)})
                channel.basic_publish(exchange='',
                                      routing_key=self.queue,
                                      body=message,
                                      properties=pika.BasicProperties(
                                          delivery_mode=2,  # make message persistent
                                      ))
                task_counter += 1

        if task_counter == 0:
            print("Nothing to do. Not waking the minions.")
        else:
            print("Released %i population orders for %s to the minions." % (task_counter,relation))
