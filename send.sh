#!/bin/bash

rsync -trlvpz --exclude-from=exclude.txt . cluster:/scratch/ks2292/run/mesh_test/
