from random import shuffle
import cv2
import io
import imageio
import numpy as np
import datajoint as dj
from stimulus import stimulus


schema = dj.schema('pipeline_movies', locals())


@schema
class QualityLabel(dj.Lookup):
    definition = """
    label         : char(4)
    ---
    numeric_label : tinyint
    """

    contents = (('good', 1), ('bad', 0))

@schema
class MovieQualityLabels(dj.Manual):
    definition = """
    -> stimulus.Movie.Clip
    ---
    -> QualityLabel
    """
    @property
    def unpopulated(self):
        return (stimulus.Movie.Clip() & AvgClipStats() & dict(movie_name='bigrun')) - self

    def load_movie(self, key):
        movie = (stimulus.Movie() * stimulus.Movie.Clip() & key).fetch1('clip')

        vid = imageio.get_reader(io.BytesIO(movie.tobytes()), 'ffmpeg')
        # convert to grayscale and stack to movie in width x height x time
        m = vid.get_length()
        movie = np.stack([vid.get_data(i) for i in range(m)], axis=0)
        return movie

    def populate(self, limit=None):
        keys = self.unpopulated.fetch(dj.key)
        shuffle(keys)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1600, 900)
        n = limit or len(keys)
        for i, key in enumerate(keys[:n]):
            movie = self.load_movie(key)
            for frame in movie:
                cv2.imshow('image', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            verdict = input('Video quality? [Good, Bad, default=g]: ')
            verdict = 'good' if verdict.lower() in ['', 'g'] else 'bad'
            key = dict(key, label=verdict)
            print('{}/{}: {}'.format(i+1, n, key))
            self.insert1(key)
        cv2.destroyAllWindows()


schema.spawn_missing_classes()