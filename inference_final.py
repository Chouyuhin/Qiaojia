import sys
import os
real_dir=os.getcwd()
sys.path.append(real_dir)



from generator.predictor_final import predictor
predictor(# input_dir="d:/hdf5_merged",
       input_hdf5="d:/hdf5_merged/2023.hdf5",
       input_model='TrainResults/models/test_trainer_080.h5',
       output_dir='PredictionResults4QJ2023raw',
       detection_threshold=0.3,                
       P_threshold=0.1,
       S_threshold=0.1, 
       number_of_plots=1000,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(20000, 3),
       number_of_cpus=4,
       batch_size=200,
       gpuid=0,
       gpu_limit=0.98)