# Autonomous_Car_Simulator with CNN and template
- Udacity Car Simulator: https://github.com/udacity/self-driving-car-sim
- Preference: https://github.com/ManajitPal/DeepLearningForSelfDrivingCars

## Train:
- Use simulator to make own train data on train mode.
- Model is in models/sim.py
- Regressor in models/regressor.py is a general version of model.
- Run run.py for training.
- logger/runs/udacity contains events for tensorboard.
- weights/udacity contains weights of model.
- config hyperparameters in run.py
...
python run.py
...

## Inference:
- Start autonomous mode.
- Run command and see the self_driving_car:
...
python drive.py
...

## Result:
![Alt Text](result/sim_car.gif)

