#!/bin/bash

### config ##############################################################
MAPS_DATASET_DIRECTORY="/home/rainer/Coding/cp/pipedream/bin/data/maps_piano"
# MAPS_DATASET_DIRECTORY=""

### leave alone, if there are no errors #################################
if [[ -z $MAPS_DATASET_DIRECTORY ]];
then
    echo "cannot link dataset; please set MAPS_DATASET_DIRECTORY variable in this script!"
fi

if [[ ! -d bin/data ]];
then
    echo "creating bin/data"
    mkdir bin/data
    echo "linking dataset directory"
    ln -s $MAPS_DATASET_DIRECTORY bin/data/maps_piano
fi

SPLIT_DIR="bin/data/maps_piano/splits"
if [[ ! -d $SPLIT_DIR ]];
then
    echo "creating split directory"
    mkdir $SPLIT_DIR
fi

if [[ ! -d $SPLIT_DIR/sigtia-4-splits ]];
then
    echo "linking configuration I splits"
    ln -s splits/sigtia-4-splits/ $SPLIT_DIR
fi

if [[ ! -d $SPLIT_DIR/sigtia-conf2-splits ]];
then
    echo "linking configuration II splits"
    ln -s splits/sigtia-conf2-splits/ $SPLIT_DIR
fi


