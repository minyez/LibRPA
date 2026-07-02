# Makefile for Sphinx documentation
# Variables for this project
DOXY_SOURCE = doxygen
DOXY_BUILDDIR = $(DOXY_SOURCE)/$(shell awk '/OUTPUT_DIRECTORY/ {print $$3}' $(DOXY_SOURCE)/Doxyfile)

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= 
SPHINXBUILD   ?= sphinx-build
DOXYSPHINX    = doxysphinx
SOURCEDIR     = .
BUILDDIR      = _build

# Targets
.PHONY: html help clean doxygen doxysphinx runtime_parameters

html: runtime_parameters doxygen doxysphinx sphinx

sphinx: runtime_parameters
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

runtime_parameters: user_guide/runtime_parameters.md

user_guide/runtime_parameters.md: user_guide/runtime_parameters.yml
	@python3 user_guide/generate_runtime_parameters.py

doxygen:
	@cd $(DOXY_SOURCE) && $(MAKE)

doxysphinx:
	@doxysphinx build $(SOURCEDIR) $(BUILDDIR)/html $(DOXY_SOURCE)/Doxyfile --doxygen_cwd $(DOXY_SOURCE)/

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

veryclean: clean
	@cd $(DOXY_SOURCE) && $(MAKE) $@
