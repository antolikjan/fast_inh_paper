# $Id: Makefile 335 2010-07-30 12:29:56Z jbednar $

# These should be set to the locations of the paper and the data from
# the runs on which it is based; also need to set the paper location in
# ./extract_figures.sh and ./get_stab_data.m
all: figures paper

figures: generate-figure-material svg-figures

generate-figure-material:
	cd CODE
	python generate_figure_material.py
	cd ..

svg-figures:
	sh build_figures.sh

paper:
	pdflatex paper.tex > /dev/null 2>&1
	bibtex paper
	pdflatex paper.tex > /dev/null 2>&1
	pdflatex paper.tex


clean:	
	rm -f *~ *.aux *.orig.bbl *.blg *.lof *.log *.lot *.pdf

figure_clean:
	rm -f SVG/Figure*/*.pdf 
	rm -f SVG/Figure*/*.eps
	rm -f SVG/Figure*/Figure*.png
	rm -f SVG/Figure*/generated_data/*

output_clean:
	rm -f paper.pdf paper.bbl 

really_clean: output_clean figure_clean clean

