.PHONY: all clean check
all: main.pdf supp.pdf check

main.pdf: main.tex sections/*.tex references.bib
	pdflatex -interaction=nonstopmode main.tex >/dev/null || true
	bibtex main >/dev/null || true
	pdflatex -interaction=nonstopmode main.tex >/dev/null || true
	pdflatex -interaction=nonstopmode main.tex

supp.pdf: supp.tex
	pdflatex -interaction=nonstopmode supp.tex >/dev/null || true
	pdflatex -interaction=nonstopmode supp.tex

check: main.pdf
	@if command -v pdftotext >/dev/null; then \
	  n=$$(pdftotext main.pdf - | grep -c "RUN REQUIRED" || true); \
	  echo "RUN REQUIRED markers in main.pdf: $$n"; \
	else echo "pdftotext not installed; grep main.log/generated instead"; fi

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc main.pdf supp.pdf
