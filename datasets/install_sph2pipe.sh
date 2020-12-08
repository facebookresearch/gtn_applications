#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates. This source code is
# licensed under the MIT license found in the LICENSE file in the root
# directory of this source tree.

# Install sph2pipe
sph_v=sph2pipe_v2.5
curl -O http://www.openslr.org/resources/3/${sph_v}.tar.gz
tar -xzvf ${sph_v}.tar.gz
cd ${sph_v} && gcc -o sph2pipe *.c -lm
cd ..
rm ${sph_v}.tar.gz
