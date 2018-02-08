# ICASSP18

Code to reproduce the experiments for the paper:

@inproceedings{kelz_icassp18,
  author    = {Rainer Kelz and Gerhard Widmer},
  title     = {Investigating Label Noise Sensitivity Of Convolutional Neural Networks For Fine Grained Audio Signal Labelling}
  booktitle = {2018 {IEEE} International Conference on Acoustics, Speech and Signal
               Processing, {ICASSP} 2018, Calgary, Alberta, Canada, April 15-20, 2018},
  year      = {2018}
}

# Installation Instructions (some assembly required)
All of the following assumes that you already properly set up your CUDA environment, and have working `numpy, scipy, theano, lasagne` installations.

- obtain the MAPS dataset (http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/)

- clone this repository
- edit `setup.sh` to let it know about the folder you extracted the dataset to
- execute `setup.sh`, it'll create some directories, and symlinks
- execute `python setup.py develop`, to install the package and all needed dependencies
- go to the `bin` directory and execute `python run.py train maps_labelnoise_conv`

- if everything went well, you'll see a printout of the exact net definitions, parameter count and a progress bar. This will be very slow in the beginning, as it needs to decode audio data / compute all (spectrogram, label) pairs. (I removed the `joblib` cache again, b/c there were some weird troubles using it. You may of course put it back into `datasources/maps_labelnoise.py`.)

- after training, go to `bin` and execute `python run.py test maps_labelnoise_conv`
- all results should be in the folders `bin/runs/maps_labelnoise_conv/<run-id>` having names `native_fps_result.pkl` and `high_fps_result.pkl`