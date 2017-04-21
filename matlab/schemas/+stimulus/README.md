# The Stimulus schema
The `stimulus` schema is a self-contained application that generates, presents, and records visual stimuli using PsychToolbox.

# How to run a stimulus from MATLAB 
Although these steps can be executed manually, they are typically automated and thus serve as the application interface for the experiment control software.

## Step 1: Prepare the screen 
```
>> stimulus.open
```

## Step 2: Generate stimulus conditions and queue trials
Stimulus trials are generated and queued by the scripts in the `+stimulus/+conf` directory.  You need to know which configuration script needs to be run.

For example, to prepare the `matisse2` stimulus, execute 
```
>> stimulus.conf.matisse2
```

While the stimulus is loaded, you will see a sequence of dots `.` and asterisks `*`, which respectively indicate whether the conditions are computed anew or are loaded from the database.  Some stimuli take a long time to compute and you might like to run the configuration before you begin the experiment.  On subsequent runs, the computed stimuli will be loaded from the database and will not take as long.

## Step 3.  Run the stimulus 
The stimulus must be run for a specific scan in the `experiment.Scan` table.  
Table `experiment.Scan` contains a dummy entry that can be used for testing.  Its primary key is `struct('animal_id', 0, 'session', 0, 'scan_idx', 0)`.  During the experiment, the correct scan identification must be provided.

The following command will run the queued stimulus trials in sequence. 
```
>> stimulus.run(struct('animal_id', 0, 'session', 0, 'scan_idx', 0))
```

## Step 4.  Interrupt and resume the stimulus
While the stimulus is playing, you can interrupt with `Ctrl+c`.  The stimulus program will handle this event, cancel the ongoing trial, and clear the screen.  To resume the stimulus, repeat the `stimulus.run` call above.  Or to queue a new set of trials, run the configuration script again.

## Step 4. Exit 
To close the stimulus program, 
```
>> stimulus.close
```


# How to run a stimulus from Python

![](erd.png)
