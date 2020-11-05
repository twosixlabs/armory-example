# Getting started with armory and armory-example

## Prerequisites

* Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* Install [docker](https://docs.docker.com/install).
* Install [python](https://www.python.org/downloads/) 3.6+ on your system.
* Recommended: create a virtual environment in which to install armory, perhaps
  with virtualenv or anaconda.

## Setting up armory

The TwoSix Armory package goes by the name `armory-testbed` on
in the pip master repository at pypi.org. Install it and check that
you got the version you think you did:
```
pip install armory-testbed
armory version
```
You can see the latest armory version in our [release
notes at GitHub](https://github.com/twosixlabs/armory/releases/latest).
Now that you have the current Armory installed, configure it and import the
examples:
```
armory configure
git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
```
The armory configure command asks some questions about where you'd like to store
the files that will be created or downloaded by armory. The armory configure
command uses the `~/.armory` directory for its defaults, although you can change
those to any location convenient to you. Demonstration output in this tutorial
will show as `~/.armory`

## The armory run command

The command:
```
armory run job-file
```
will run the evaluation job and display logs to you. When the job completes the
container it creates will be automatically removed. Results from the evaluation
job can be found in your output directory selected in the `armory configure`
step. You can find sample job-files in the `armory-example` directory in the
`official_scenario_configs` and `example_scenarios` directories.

Using one of the provided jobs, run
```
armory run official_scenario_configs/so2sat_baseline.json
```
this will take up to an hour to run depending on your network connection
and available processor, printing status information the whole time. The job
can be interrupted without harm by typing `ctrl-c`. If you do interrupt a
running job, you will lose any computational work done so far.

There are four main phases to the armory job execution:
 1. pulling the docker images needed to build a container and building it
 2. downloading the datasets needed to run the job
 3. training the model
 4. evaluating the model

and the extremely excerpted console output from these look like:
```
2020-11-04 21:09:43 omen armory.utils.docker_api[788552] INFO 45d437916d57: Download complete
armory.data.utils[6] INFO Downloading dataset: cifar10...
82% 111M/135M [00:08<00:01, 20.4MB/s]
armory.scenarios.image_classification[6] INFO Fitting estimator on clean train dataset...
Epoch 19/20:  36% 278/781 [00:01<00:02, 222.63it/s]
art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch[6] INFO Success rate of attack: 67.19%
Attack:  38% 60/157 [01:41<02:41,  1.66s/it]
INFO Saving evaluation results saved to <output_dir>/ImageClassificationTask_1604542702.json
Scenario has finished running cleanly
```
Don't be concerned with the details here, it is just to give you a taste of
the thousands of lines of logging that are generated. The last two lines are
important.

> TODO: check with neal that the output/iso-8601-datestamp directory ought be
explained

## Scenario output

When the `armory run` job completes, results will be placed your
`~/.armory/outputs` directory, `ImageClassificationTask_1604542702.json` in
the example above.

The
output json will be the sole file in the directory. The output json contains the
version of armory it was run on, the job it was run with, and a results
dictionary containing the relevant
[metrics](https://armory.readthedocs.io/en/latest/metrics/) from the run, and a
timestamp.

# How to make an armory job

An armory job consists of a job-file which describes the model evaluation
along with files specified in that job. Rather than constructing a job from
scratch, it is easier to copy an existing job and modify it. We'll run
through an example here.

### Model updates

We'll start with the [so2sat_baseline][so2base] job. The so2sat_baseline
describes a multimodal classification problem against the SO2Sat dataset.
For convenience, we'll also use a pre-built [weights file][weights].
Outside of this demo, you'd probably build your own weights via model
training.

For armory to evaluate a neural network model, it needs to be wrapped
with [Adversarial Robustness Toolbox][art] estimator. These wrappers
enable the ART suite of attacks as well as custom attacks to be run
against the model. Starting from the [unwrapped model][unwrapped]
we'll add a wrapping function
```
from art.classifiers import PyTorchClassifier
…
def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    model = make_so2sat_model(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
…
```
You can see the [completed wrapped model][wrapped] file for the rest of that
python function.

  [so2base]: https://github.com/twosixlabs/armory-example/blob/master/official_scenario_configs/so2sat_baseline.json
  [weights]: https://armory-public-data.s3.us-east-2.amazonaws.com/model-weights/so2sat_split_weights.model
  [art]: https://github.com/Trusted-AI/adversarial-robustness-toolbox
  [unwrapped]: model_to_integrate/model/so2sat_split_unintegrated.py
  [wrapped]: model_to_integrate/model/so2sat_split.py

As seen in the integrated [model](model_to_integrate/model/so2sat_split.py),
to integrate, we add a method that takes arguments of ```model_kwargs, wrapper_kwargs, weight_path``` which
returns the model with the weights from ```weight_path``` (the full path to weights file(s))
loaded and returns a wrapped version of the model.

### Weights file transfer
The weights file need to be copied to the config's <saved_model_dir>, by default at ```~/.armory/saved_models```.
```cp so2sat_split_weights.model ~/.armory/saved_models```

### Config updates
For more details on Armory configurations, please refer to https://armory.readthedocs.io/en/latest/configuration_files.
The model can be integrated by adapting the official
[config](https://github.com/twosixlabs/armory-example/blob/master/official_scenario_configs/so2sat_baseline.json).

We copy the original config and update the following fields:
model: We update the model to refer to the path of the example_model, in this case, the module is
```model_to_integrate.model.so2sat_split```. If the method that returned the model had a different
named than ```get_art_model```, we would update the ```name``` field. Note: the model does not need to
reside in the same repo as ```armory-example```. We can add external Github repos as described in the
External repo [documentation](https://armory.readthedocs.io/en/stable/external_repos/).

We update the reference to the weights files to refer to the weights file so2sat_split_weights.model. Note
that we simply use the name of the weights file, not the full path.

Finally, since the official scenario config was a Keras model and we are using a Pytorch model, we update
the docker image to be the Pytorch image.

The final file with all the changes is available for [reference](example_scenario_configs/integrate_so2sat_ref.json).

### Checking integration

We can check integration quickly by running ```armory run --check <path/to/new/config>```. This will run
armory on a single batch of inputs, and without training (except for the poisoning scenario, which requires
training for integration).

## Common errors
`pip install armory-testbed` returns “Could not find a version that satisfies the requirement armory-testbed”
*	This is likely because you are trying to install armory with a python version prior to 3.6. Ensure your python
environment version is 3.6 or greater.

armory download <path/to/config.json> returns a KeyError: 'dataset_name'
* Armory download does not take regular configs as arguments. This is a feature we are exploring adding, but for now
it only takes download configs such as https://github.com/twosixlabs/armory-example/blob/master/scenario_download_configs/scenarios-set2.json
as arguments.

## Additional reading:
Extended documentation are found here: https://armory.readthedocs.io/en/latest/

Primary Armory GitHub repo: https://github.com/twosixlabs/armory

# Stuff that Matt excised that might want to be put back

> In the context of a getting started tutorial this seems to be unnecessary
> complexity since `run` will load stuff anyway.

Before running an armory job, you may want to pre-download all of the
model weights and datsets by running:
```
armory download scenario_download_configs/scenarios-set2.json
```
from the root of the armory-example repo. This process is relatively slow, so we
recommend running it overnight if feasible. Note that each job that is run
automatically will download the relevant weights and datasets when it is run
with `armory run` in any case.

---

> If no-docker can be taken out of the happy path of the tutorial it reads more
> cleanly.

> From prerequisites

See the section below on no-docker mode if your environment does not permit docker.



### Using no-docker mode


If you work in an environment where docker is not permitted, you can run armory
in
[no-docker](https://armory.readthedocs.io/en/latest/docker/#running-without-docker)
mode.

If you use no-docker mode, you'll need to install additional packages for
armory. These additional packages are listed in
[host-requirements.txt](https://github.com/twosixlabs/armory/blob/master/host-requirements.txt)
Depending on which framework you use (TensorFlow1, TensorFlow2, PyTorch,
etc.) some entries will need to be uncommented as described in that file.
Then run
```
pip install -r host-requirements.txt
```

---

> again, to keep the tutorial happy path clean I took out

### Debugging an armory run
To access the running job with tools such as [pdb](https://docs.python.org/3/library/pdb.html),
armory can be run in interactive mode:
```
armory run job-file --interactive
```
which starts the container specified in the job-file, copies the job-file into
the container, and shows you the commands to attach to the running container.
Similar to non-interactive mode, job results are placed in the output directory.
To later close the interactive container, type `ctrl-c` in the terminal where
this `armory run` command was started.
