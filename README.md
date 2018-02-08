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

- obtain the MAPS dataset (http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/)

- clone this repository
- edit `setup.sh` to let it know about the folder you extracted the dataset to
- execute `setup.sh`, it'll create some directories, and symlinks
- execute `python setup.py develop [--user]` [--user] is optional