#!/bin/bash

for x in ./SVG/FigureORTuningSummary/FigureORTuningSummary.svg 
  do
  dir=`dirname $x`
  base=`basename $x .svg`
  pwd
  cd ${dir}; inkscape ${base}.svg --export-area-drawing --export-dpi=600 -A ${base}.eps; cd ../../
  pwd
  cd ${dir}; inkscape ${base}.svg --export-area-drawing --export-dpi=600 -e ${base}.png; cd ../../
  epstopdf ${dir}/${base}.eps 
  #xpdf ${dir}/${base}.pdf &
done