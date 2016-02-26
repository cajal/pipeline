from contextlib import contextmanager
import pika
import pickle
from . import SIMPLE_POPULATOR


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

    def populate(self, relation, restrictions=None):
        with self.connection() as channel:
            message = pickle.dumps({'relation': relation, 'restrictions': restrictions})
            channel.basic_publish(exchange='',
                                  routing_key=self.queue,
                                  body=message,
                                  properties=pika.BasicProperties(
                                      delivery_mode=2,  # make message persistent
                                  ))

            if restrictions is None:
                print(" Ordered a minion to populate %s" % (relation,))
            else:
                print(" Ordered a minion to populate %s with restrictions %s" % (relation, str(restrictions)))
