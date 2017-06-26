# License
This software is published under the [LGPLv3 license](LICENSE.txt).

# Pipeline Tutorial
1.  [Intro](jupyter/tutorial/pipeline_intro.ipynb)
1.  [Big Picture](jupyter/tutorial/big_picture.ipynb)
1.  [pipeline.experiment](jupyter/tutorial/pipeline_experiment.ipynb)
1.  [pipeline.vis](jupyter/tutorial/pipeline_vis.ipynb)
1.  [pipeline.preprocess](jupyter/tutorial/pipeline_preprocess.ipynb)
1.  [Synchronization](jupyter/tutorial/pipeline_synchronization.ipynb)
1.  [Processing Monet stimulus](jupyter/tutorial/pipeline_prepare_monet.ipynb)
1.  [Processing movie stimulus](jupyter/tutorial/pipeline_prepare_movie.ipynb)
1.  [pipeline.tuning](jupyter/tutorial/pipeline_tuning.ipynb)
1.  [Eyetracking traces](jupyter/tutorial/pipeline_eyetracking.ipynb)

# Versioning
Version tags (in MAJOR.MINOR format) are added to your current commits as
```sh
git tag -a reso-v1.3
```
and pushed to the remote repository as
```
git push origin reso-v1.3
```
The major version of the tag corresponds to changes in the pipeline that will modify results; this is also the number stored under version in the database. Minor versions can be performance improvements or bug fixes.

Whenever we change major versions, we will make a release with the latest working version of the previous pipeline, so for instance when we upgrade to reso-v2.0, the latest reso-v1.x will be released as reso-v1.

