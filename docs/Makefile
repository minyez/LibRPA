DOXYFILE = doxygen/Doxyfile
DOXY_OUTPUT_DIRECTORY = build_doxy

.PHONY: default doxy clean

default: $(DOXY_OUTPUT_DIRECTORY)

$(DOXY_OUTPUT_DIRECTORY): $(DOXYFILE)
	doxygen $<

clean:
	rm -rf $(DOXY_OUTPUT_DIRECTORY)
