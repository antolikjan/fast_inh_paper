#!/bin/bash

for x in ./SVG/Figure1/figure1.svg  ./SVG/Figure2/figure2.svg ./SVG/Figure3/figure3.svg  ./SVG/Figure4/figure4.svg ./SVG/Figure5/figure5.svg ./SVG/FigureModelArchitecture/drawing.svg ./SVG/FigureModelVariants/drawing.svg ./SVG/FigureLateralInteractions/drawing.svg 
  do
  dir=`dirname $x`
  base=`basename $x .svg`
  pwd
  cd ${dir}; inkscape ${base}.svg --export-area-drawing --export-dpi=600 -e ${base}.png; cd ../../
  cd ${dir}; inkscape ${base}.svg --export-area-drawing --export-dpi=600 -e ${base}.tif; cd ../../
done
