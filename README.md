# armory-example
This repository is meant to be an example of how to run 
[ARMORY](https://github.com/twosixlabs/armory) adversarial defense evaluations.

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

We've added an example Scenario (audio classification using spectrograms), as well as
some custom configurations for different using different attacks and defenses using 
ART modules.