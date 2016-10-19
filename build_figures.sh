#!/bin/bash

for x in ./SVG/Figure1/figure1.svg  ./SVG/Figure2/figure2.svg 
  do
  dir=`dirname $x`
  base=`basename $x .svg`
  pwd
  cd ${dir}; inkscape ${base}.svg --export-area-drawing --export-dpi=600 -e ${base}.png; cd ../../
done