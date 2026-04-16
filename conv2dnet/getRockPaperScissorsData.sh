#!/bin/bash
curl -L -o ${PWD}/rockpaperscissors.zip\
  https://www.kaggle.com/api/v1/datasets/download/drgfreeman/rockpaperscissors

mkdir rockpaperscissors
unzip ${PWD}/rockpaperscissors.zip -d ${PWD}/rockpaperscissors
