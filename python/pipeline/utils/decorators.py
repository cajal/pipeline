import datajoint as dj
import inspect
import warnings
import os
import git
import datetime

def gitlog(cls):
    """
    Decorator that equips a datajoint class with an additional datajoint.Part table that stores the current sha1,
    the branch, the date of the head commit,and whether the code was modified since the last commit,
    for the class representing the master table. Use the instantiated version of the decorator.
    Here is an example:
    .. code-block:: python
       :linenos:
        import datajoint as dj
        from djaddon import gitlog
        schema = dj.schema('mydb',locals())
        @schema
        @gitlog
        class MyRelation(dj.Computed):
            definition = ...
    """
    class GitKey(dj.Part):
        definition = """
        ->master
        ---
        sha1        : varchar(40)
        branch      : varchar(50)
        modified    : int   # whether there are modified files or not
        head_date   : datetime # authored date of git head
        """

    def log_key(self, key):
        key = dict(key) # copy key
        path = inspect.getabsfile(cls).split('/')
        for i in reversed(range(len(path))):
            if os.path.exists('/'.join(path[:i]) + '/.git'):
                repo = git.Repo('/'.join(path[:i]))
                break
        else:
            raise KeyError("%s.GitKey could not find a .git directory for %s" % (cls.__name__, cls.__name__))
        sha1, branch = repo.head.commit.name_rev.split()
        modified = (repo.git.status().find("modified") > 0) * 1
        if modified:
            warnings.warn('You have uncommited changes. Consider committing the changes before running populate.')
        key['sha1'] = sha1
        key['branch'] = branch
        key['modified'] = modified
        key['head_date'] = datetime.datetime.fromtimestamp(repo.head.commit.authored_date)
        self.GitKey().insert1(key, skip_duplicates=True, ignore_extra_fields=True)
        return key


    cls.GitKey = GitKey
    cls.log_git = log_key

    return cls
