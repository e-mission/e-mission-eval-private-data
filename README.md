This repository contains ipython notebooks for the evaluation of the e-mission
platform.  These notebooks re-use code from the e-mission-server codebase, so
it needs to be included while running them.

### Running.

1. Install the e-mission server, including setting it up
    https://github.com/e-mission/e-mission-server

1. Set the home environment variable

    ```bash
    $ export EMISSION_SERVER_HOME=<path_to_emission_server_repo>
    ```

     To verify, check the environment variables using 

        $ env

    and ensure ```ENV_SERVER_HOME``` is present in the list and has the right path (as mentioned above).
    
1. If you haven't setup before, set up the evaluation system

    ```bash
    $ source setup.sh
    ```
1. If you have, activate

    ```bash
    $ source activate.sh
    ```

1. Access the visualizations of interest and copy the config over. The `<eval_folder>` mentioned below can be any folder containing notebooks and/or .py files for visualisation or other purposes. E.g. : `TRB_label_assist` is one such folder.


```bash
$ cd <eval_folder>
$ cp -r ../conf .
```

1. Start the notebook server

```bash
$ ../bin/em-jupyter-notebook.sh
```

### Loading data

- To get the data for the notebooks to run on, look at the dataset listed at
  the top of the notebook, and request the data for research purposes using 
    https://github.com/e-mission/e-mission-server/wiki/Requesting-data-as-a-collaborator

- Assuming that your data is in the "mongodump" format, this repository has a helper script to load the data directly into the database.  
    - Navigate to the `e-mission-eval-private-data/` directory and start the docker environment
        ```bash
            $ docker-compose -f docker-compose.dev.yml up
        ```
    - In another terminal, again navigate to the repository. Using the script provided, load the mongodump into docker
        ```bash
        $ bash bin/load_mongodump.sh <mongodump_file.tar.gz>
        ```
    - Depending on the size of the mongodump, the loading step may take quite a long time (up to _several hours_).  For more details on how to speed up this process, please refer to the data request documentation [here](https://github.com/e-mission/e-mission-server/wiki/Requesting-data-as-a-collaborator).
### Cleaning up

After completing analysis, tear down

```
$ source teardown.sh
```

### Checking in notebooks

Note that all notebooks checked in here are completely public. All results included in them can be viewed by anybody, even malicious users. 
Therefore, you need to split your analysis into two groups:
- *aggregate only*: results are not specific for a single user. The scripts in such notebooks should not include uuids, and should use the aggregate timeseries instead of the default timeseries.
   - example: number of walking and biking trips over all users in the control group
- *individual analyses*: results are specific for a single user. The scripts in such notebooks can include uuids, and potentially even user emails or tokens.
   - example: varation in walking and biking trips over time for user `uuid1`

Notebooks that include aggregate analyses can be checked in with outputs included. This is because it is hard to tease out the contributions by individuals to the aggregate statistics, and so the chances of leaking information are low. However, notebooks that include individual analyses should be checked in after deleting all outputs (Kernel -> Restart and clear output).

|              | Aggregate results | Individual results |
|--------------|--------------|--------------|
| with outputs |     Y        |     **N**    |
| after clearing outputs | Y  |     Y        | 
