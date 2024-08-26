# CODA: Combined Optimization of Dynamics and Assimilation with End-to-End Learning on Sparse Observations
A novel method for training data assimilation networks directry from sparse and noisy observations, tuning free parameters of a simulator and learning effect of unresolved processes.

## Data generation
Lorenz'96 training data can be generated using generation script from [mdml-tools](https://codebase.helmholtz.cloud/m-dml/mdml-tools/-/blob/main/mdml_tools/scripts/generate_lorenz_data.py).


## Running experiments
We use hydra to manage config files for our experiments. To see all availabable parameters and flags refere to corresponding experiment configs in `conf/experiment`. All parameters can be overwritten from command line.

### Data assimilation
Example for running *data assimilation* experiment on *Lorenz' 96 one-level* data:
```bash
python main.py +experiment=data_assimilation output_dir_base_path="path/where/to/save" datamodule.path_to_load_data="path/to/load/l96/simulation.h5" l96forcing=8.0 observation_model.additional_noise_std=1.0 observation_model.random_mask_fraction=0.75 rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111
```

### Parameter tuning
Example for running *parameter tuning* experiment on *Lorenz'96 one-level* data:
```bash
python main.py +experiment=parameter_tuning output_dir_base_path="path/where/to/save" datamodule.path_to_load_data="path/to/load/l96/simulation.h5" observation_model.additional_noise_std=1.0 observation_model.random_mask_fraction=0.75 rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111
```

### Parametrization learning
Example for running *pretraining data assimilation network* experiment on *Lorenz'96 two-level* data:
```bash
python main.py +experiment=pretrain_data_assimilation output_dir_base_path="path/where/to/save" datamodule.path_to_load_data="path/to/load/l96/simulation.h5" l96forcing=10.0 observation_model.additional_noise_std=1.0 observation_model.random_mask_fraction=0.75 rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111
```

Example for running *parametrization learning* experiment on *Lorenz'96 two-level* data:
```bash
python main.py +experiment=parametrization_tuning output_dir_base_path="path/where/to/save" assimilation_network_checkpoint="path/to/pretrained/da/network/checkpoint" datamodule.path_to_load_data="path/to/load/l96/simulation.h5" l96forcing=10.0 observation_model.additional_noise_std=1.0 observation_model.random_mask_fraction=0.75 rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111
```
