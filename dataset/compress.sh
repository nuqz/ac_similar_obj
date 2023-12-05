#!/bin/bash

FILENAME=$1

tar -czvf ./dataset/$FILENAME.tar.gz ./dataset/challenges ./dataset/solutions
