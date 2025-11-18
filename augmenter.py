import Augmentor
import os

input_dir = "/home/aryan/Desktop/Adl_assignment_1/masked_dataset/train"
subjects = os.listdir(input_dir)

for s in sorted(subjects):
    if os.path.isdir(os.path.join(input_dir, s)):
        
        p = Augmentor.Pipeline(os.path.join(input_dir, s))

        p.random_distortion(probability=0.5, grid_width=8, grid_height=8, magnitude=4)

        p.sample(100)