#!/bin/bash

# Activate environment
source /home/nbcommon/anaconda3_501/bin/activate

# Set up proxy
http_proxy=http://webproxy:3128
https_proxy=http://webproxy:3128
export http_proxy 
export https_proxy

# pip
pip install -r /home/nbuser/library/requirements.txt
