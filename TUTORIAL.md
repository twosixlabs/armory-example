# Getting started with armory and armory-example

## Prerequisites

* Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* Install [python](https://www.python.org/downloads/) 3.6+ on your system.
* Install [docker](https://docs.docker.com/install).

Like many python applications, we recommend creating a virtual environment in
which to install armory, perhaps with virtualenv or anaconda.

If your local rules prohibit docker execution, there is limited
support described in The Armory Documentation under [Running without
docker][nodocker].

  [nodocker]: https://armory.readthedocs.io/en/latest/docker/#running-without-docker

## Setting up armory

The TwoSix Armory package goes by the name `armory-testbed` on
in the pip master repository at pypi.org. Install it and check that
you've got the version you think you did:
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
files that will be created or downloaded by armory. The armory configure
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

> TODO: is there a explainable difference between the two directories?

To evaluate one of the provided scenarios, run
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
Don't be concerned with the details, it is just to give you a taste of
the thousands of lines of logging that are generated. The last two lines are
important.


## Scenario output

When the armory job completes, results will be put in a date-stamped
subdirectory in your `~/.armory/outputs` directory. For the example
run above the output file
```
~/.armory/outputs/2020-11-05T020942.176390/ImageClassificationTask_1604542702.json
```
was created.

The output json contains the
version of armory it was run on, the job it was run with, and a results
dictionary containing the relevant
[metrics](https://armory.readthedocs.io/en/latest/metrics/) from the run, and a
timestamp.

# How to make a new armory job

An armory job consists of an armory scenario configuration which describes the model
evaluation along with files by that that job. Rather than constructing a scenario from
scratch, it is easier to copy an existing scenario and modify it. We'll run through an
example here.

We'll start with the [so2sat_baseline][so2base] job. The so2sat_baseline describes a
multimodal classification problem against the SO2Sat dataset. For convenience, we'll
also use a pre-built weights file. Outside of this demo, you'd probably build your own
weights via model training.

We copy that original job to `my_so2sat.json` and update specifications
within it.
```
cp official_scenario_configs/so2sat_baseline.json my_so2sat.json
```
In that file there are references to:

  1. the `"module"` and `"name"` lines which point to the python module and method which
     provide an ART model, and
  2. the `"weights_file"` line which tells armory where to obtain model weights.

We'll change the referents of these in turn and then alter the `my_so2sat.json` job file
to match them.


### Model updates

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
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(14, 32, 32),
        nb_classes=17,
        **wrapper_kwargs,
    )
    return wrapped_model
```
As shown, we adapt the `get_art_model` method with the weights from `weights_path`
loaded and returns a wrapped version of the model.

An already adapted version of the model file is provided in the
[so2sat_split.py][wrapped] in the `model_to_integrate` directory.

  [so2base]: https://github.com/twosixlabs/armory-example/blob/master/official_scenario_configs/so2sat_baseline.json
  [weights]: https://armory-public-data.s3.us-east-2.amazonaws.com/model-weights/so2sat_split_weights.model
  [art]: https://github.com/Trusted-AI/adversarial-robustness-toolbox
  [unwrapped]: model_to_integrate/model/so2sat_split_unintegrated.py
  [wrapped]: model_to_integrate/model/so2sat_split.py

### Weights file transfer

For this demo we need to download a weights file from armory-public file storage on
Amazon S3 and place them in `~/.armory/saved_models`.
```
curl -O https://armory-public-data.s3.us-east-2.amazonaws.com/model-weights/so2sat_split_weights.model
cp so2sat_split_weights.model ~/.armory/saved_models
```
> TODO: Matt needs to find out why weights would **not** have been retrieved by `armory
> run`


### Updating the scenario configuration file

Now that we have modified the model and installed the weights, we need to alter a
evaluation config file to incorporate them. We integrate these into the scenario
configuration we copied into `my_so2sat.json` earlier

First, we update the model to refer to the path of the example_model, in this
case, the module is `model_to_integrate.model.so2sat_split`. Because
we named our new model function `get_arg_model` the model name field
remains unchanged.

Next we change weights_files to refer to the weights file we imported
above. This yields the line:
```
"weights_file": "so2sat_split_weights.model"
```
in our job configuration file. Here we only specify the name of the weights
file, not the full path; armory knows to look for it in the
`~/.armory/saved_models` directory.

Finally, since the official scenario config was a TensorFlow.Keras model and we are now
using a PyTorch model, we update the docker image to the PyTorch flavor yielding the
line
```
"docker_image": "twosixarmory/pytorch:0.12.1"
```
in the sysconfig block.

An already adapted [scenario configuration file][integrated] is provided. It contains
the changes we've made. You can copy the updated scenario if you want
```
cp example_scenario_configs/integrate_so2sat_ref.json my_so2sat.json
```
The Armory documentation has full details of the [configuration file format][configs].

  [configs]: https://armory.readthedocs.io/en/latest/configuration_files
  [external-repo]: https://armory.readthedocs.io/en/stable/external_repos/
  [integrated]: example_scenario_configs/integrate_so2sat_ref.json

> TODO: this section could make a lot more sense if there were symmetry between the
> scenario config and model files in before and after versions. Something like
> model_{before,after}.py and config_{before,after}.json. I don't know how much the
> current names represent some armory convention.

> TODO: should the scenario _description field be changed? It isn't really a baseline
> config any longer and it might be good to show performers that they _should_
> try to keep the name meaningful

### Check integration and run

You can check that the new integration works by running
```
armory run --check my_so2sat.json
```
This will run armory on a single batch of inputs and without training. After this
check runs without errors you can evaluate the new job:
```
armory run my_so2sat.json
```
Like all armory runs, this will leave its results in the `~/.armory/out`.

# Appendices

Here are additional details that some new users of Armory have
found useful.

## Downloading the datasets in advance

Before running an armory job, you may want to pre-download all of the model
weights and datsets by running:
```
armory download scenario_download_configs/scenarios-set2.json
```
in your armory-example directory. Depending on your network connection you might
want let this run overnight or at least across a long lunch break. This step is
optional. Each job will automatically download the relevant weights and datasets
when it is run with `armory run` if the needed files are not pre-downloaded.


## Common errors

`pip install armory-testbed` returns “Could not find a version that satisfies
the requirement armory-testbed”

  * This is likely because you are trying to install armory with a python
    version prior to 3.6. Ensure your python environment version is 3.6 or
    greater.

`armory download job-file`  returns a KeyError: 'dataset_name'

  * Armory download does not take regular configs as arguments. This is a
    feature we are exploring adding, but for now it only takes download configs
    such as
    https://github.com/twosixlabs/armory-example/blob/master/scenario_download_configs/scenarios-set2.json
    as arguments.

### Debugging an armory run

To access the running job with tools such as
[pdb](https://docs.python.org/3/library/pdb.html), armory can be run in
interactive mode:
```
armory run job-file --interactive
```
which starts the container specified in the job-file, copies the job-file into
the container, and shows you the commands needed to attach to the running
container. Similar to non-interactive mode, job results are placed in the output
directory. To later close the interactive container, type `ctrl-c` in the
terminal where this armory run command was started.

## Additional reading:

The complete Armory documentation: https://armory.readthedocs.io/en/latest/

The Armory GitHub repository: https://github.com/twosixlabs/armory

# stuff matt pulled out of the happy path again

Note: the model does not need to reside in the same repo as
`armory-example`. We can add external Github repos as described in the [External
repo documentation][external-repo]).

---
