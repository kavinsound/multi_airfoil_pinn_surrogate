#!/bin/bash

rsync -trlvpz --exclude-from=.exclude . cluster:/scratch/ks2292/run/mesh_test/
