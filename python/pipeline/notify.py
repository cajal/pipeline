import datajoint as dj
from datajoint.jobs import key_hash

from .experiment import Person
schema = dj.schema('pipeline_notification', locals())

@schema
class SlackConnection(dj.Manual):
    definition = """
    # slack domain and api key for notification

    domain         : varchar(128) # slack domain
    ---
    api_key        : varchar(128) # api key for bot connection
    """

@schema
class SlackUser(dj.Manual):

    definition = """
    # information for user notification

    -> Person
    ---
    slack_user          : varchar(128) # user on slack
    -> SlackConnection
    """

    def notify(self, message=None, file = None, file_title=None, file_comment=None):
        if self:
            try:
                from slacker import Slacker
            except ModuleNotFoundError:
                pass
            else:
                api_key, user = (self * SlackConnection()).fetch1('api_key','slack_user')
                s = Slacker(api_key, timeout=60)
                if message: # None or ''
                    s.chat.post_message('@' + user, message, as_user=True)
                if file is not None:
                    s.files.upload(file_=file, channels='@' + user,
                                   title=file_title, initial_comment=file_comment)

def temporary_image(array, key):
    import matplotlib
    matplotlib.rcParams['backend'] = 'Agg'
    import matplotlib.pyplot as plt
    import seaborn as sns
    with sns.axes_style('white'):
        plt.matshow(array, cmap='gray')
        plt.axis('off')
    filename = '/tmp/' + key_hash(key) + '.png'

    plt.savefig(filename)
    sns.reset_orig()
    return filename
