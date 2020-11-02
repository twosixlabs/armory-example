# Getting started with armory and armory-example

## Prerequisites

* Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
) on your system.
* Install [docker](https://docs.docker.com/install) on your system. Note: for academic setups and 
other environments that do not permit docker, Armory has a [no-docker](https://armory.readthedocs.io/en/latest/docker/#running-without-docker) mode.
* Install [python](https://www.python.org/downloads/) 3.6+ on your system.
*Recommended: create a virtual environment in which to install armory, perhaps with virtualenv or anaconda.

## Setting up armory
```
pip install armory-testbed
armory configure # Update armory directory paths, other config settings. Recommend ~/git for git_
mkdir -p ~/git/twosixlabs
cd ~/git/twosixlabs
git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
armory version # Confirm you are using the latest based on release notes at https://github.com/twosixlabs/armory/releases
```

For users that rely on no-docker mode, please refer to [host-requirements.txt](https://github.com/twosixlabs/armory/blob/master/host-requirements.txt)
for the latest requirements for armory (note that the requirements differ depending on the framework that is used, e.g. Tensorflow or Pytorch).

Before running a config, it is possible to download all model weights and datsets by running:
```armory download scenario_download_configs/scenarios-set2.json```
from the root of the armory-example repo. This process is relatively slow, so we recommend running it
overnight if feasible. Note that each config that is run automatically will download the relevant
weights and datasets when it is run with ```armory run``` in any case. 

## Running a scenario
```armory run```

```armory run <path/to/config.json>.``` This will run a configuration file end to end.
Stdout and stderror logs will be displayed to the user, and the container will be removed
gracefully upon completion. Results from the evaluation can be found in your output
directory. Example configs are found in the ```armory-example``` repo under the directories
 ```official_scenario_configs``` and ```example_scenarios```.


```armory run <path/to/config.json> --interactive```. This will launch the framework-specific
container specified in the configuration file, copy the configuration file into the
container, and provide the commands to attach to the container in a separate terminal and run
the configuration file end to end while attached to the container. Similar to non-interactive
mode, results from the evaluation can be found in the output directory. To later close
the interactive container simply run CTRL+C from the terminal where this command was ran.

## Scenario output

Armory's log output will denote the directory name with the results output. The output json will be
the sole file in the directory. The output json contains the version of armory it was run on, the 
config it was run with, and a results dictionary containing the relevant
[metrics](https://armory.readthedocs.io/en/latest/metrics/) from the run, and a timestamp. 

## Integrating a model to work with armory

### Model updates

The following example integrates a Pytorch model with Armory. It modifies the official scenario
[config](https://github.com/twosixlabs/armory-example/blob/master/official_scenario_configs/so2sat_baseline.json)
for the multimodal classification problem on the So2Sat dataset and a baseline
[model](model_to_integrate/model/so2sat_split_unintegrated.py) that is not integrated
with ART or Armory. The weights file is available from S3 at
https://armory-public-data.s3.us-east-2.amazonaws.com/model-weights/so2sat_split_weights.model
though a typical users workflow would work with a local weights file from model training.

Neural network models to be integrated with Armory need to by wrapped with
[ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) estimators. These
wrappers enable the ART suite of attacks as well as custom attacks to be run against the
models. For the example [model](model_to_integrate/model/so2sat_split_unintegrated.py),
this requires the imporg ```from art.classifiers import PyTorchClassifier``` to wrap the model
with an ART PytorchClassifier. As seen in the integrated [model](model_to_integrate/model/so2sat_split.py),
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