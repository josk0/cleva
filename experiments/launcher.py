"""Methods that instantiate the runners.run class: for experiments and parameter tuning"""

import mlxp

@mlxp.launch(config_path='./configs')
def my_task(ctx: mlxp.Context)->None:

  # Displaying user-defined options from './configs/config.yaml
  print(ctx.config)

  # Logging information in log directory created by MLXP: (here "./logs/1" )
  for i in range(ctx.config.num_epoch):
     print(f"Logging round: {i}")
     ctx.logger.log_metrics({"epoch":i}, log_name="Quickstart")



if __name__ == "__main__":
  my_task()