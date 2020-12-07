#!/bin/bash

# Install sph2pipe
sph_v=sph2pipe_v2.5
curl -O http://www.openslr.org/resources/3/${sph_v}.tar.gz
tar -xzvf ${sph_v}.tar.gz
cd ${sph_v} && gcc -o sph2pipe *.c -lm
cd ..
rm ${sph_v}.tar.gz
