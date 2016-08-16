# Jupyter Server with Pipeline

```
sudo docker run --rm -it -e DJ_PASS='MYSQL_SERVER_PWD' -e DJ_USER='MYSQL_USER' \
                         -e DJ_HOST='IP_OF_HOST' ninai/pipeline:jupyter
```

# Populate Tables

Run populate on e.g. `preprocess.Spikes`:

```
sudo docker run --rm -it -e DJ_PASS='MYSQL_SERVER_PWD' -e DJ_USER='MYSQL_USER' \
                         -e DJ_HOST='IP_OF_HOST' ninai/pipeline populate pipeline.preprocess.Spikes
```

Run it as a daemon in the background:

```
sudo docker run -d -e DJ_PASS='MYSQL_SERVER_PWD' -e DJ_USER='MYSQL_USER' \
                   -e DJ_HOST='IP_OF_HOST' ninai/pipeline populate pipeline.preprocess.Spikes -d
```

# Eye Tracking

## Draw regions of interest for Tracking (Linux only)

```
sudo docker run -it --rm -e DJ_PASS='MYSQL_SERVER_PWD' -e DJ_USER='MYSQL_USER' \
                    -e DJ_HOST='IP_OF_HOST' -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \ 
                    -v /mnt:/mnt ninai/pipeline populate pipeline.preprocess.Eye 
```

When you cancel the population, you'll have one error job in the `preprocess.schema.jobs` table that you'll need to delete. 
Its `table_name` is probably called `preprocess._eye`.

## Plot tracking results for a particular animal

```
from pipeline import preprocess
(preprocess.EyeTracking() & dict(animal_id=8623)).plot_traces('./')
```

## Display video in a particular frame range

```
(preprocess.EyeTracking() & dict(animal_id=8623, scan_idx=13)).show_video(24000,27000)  
```