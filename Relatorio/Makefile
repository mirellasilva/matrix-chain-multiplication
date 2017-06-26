TCC = tcc
REMOVE = $(TCC).aux $(TCC).bbl $(TCC).blg $(TCC).dvi $(TCC).idx $(TCC).log $(TCC).toc this.aux thisen.aux autocitacaopt.aux autocitacaoen.aux *~

main: $(TCC).tex
	pdflatex $(TCC).tex
	bibtex $(TCC).aux
	bibtex autocitacaopt.aux
	bibtex autocitacaoen.aux
	pdflatex $(TCC).tex > /dev/null
	pdflatex $(TCC).tex > /dev/null		
	evince $(TCC).pdf &

clean: 
	rm $(REMOVE)
	echo ("Cleaned!")
