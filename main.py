from flight_maneuvers.data_module import FlightTrajectoryDataModule
from flight_maneuvers.utils import plot_scenario_3d


ex_scenario = FlightTrajectoryDataModule(sampling_period=30).train_dataloader().dataset[0]

plot_scenario_3d(ex_scenario)
