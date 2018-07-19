#!/bin/bash

conda env create -f /home/nbuser/library/environment.yml
source activate em-apps-environment

pip install -r /home/nbuser/library/requirements.txt
