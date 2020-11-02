# armory-example
This repository is meant to be an example of how to run 
[Armory](https://github.com/twosixlabs/armory) adversarial defense evaluations.
For more information about Armory, please refer to the official
[docs](https://armory.readthedocs.io/en/latest/). A tutorial on using Armory and
armory-example can be found [here](TUTORIAL.md).

### Official Scenarios
Configuration files for official scenarios are provided. 
See [official_scenario_configs directory](official_scenario_configs)

To run (after armory has been installed):
```
armory run official_scenario_configs/cifar10_baseline.json
```

### Custom Models and Scenarios
Since armory mounts your current working directory inside of evaluation docker 
containers, users are able to run their own custom Scenarios, Models, etc.

We've added some custom configurations for using different attacks and 
defenses using ART modules.