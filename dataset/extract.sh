#!/bin/bash

FILENAME=$1

tar -xzvf ./dataset/$FILENAME.tar.gz -C ./dataset --strip-components=2
