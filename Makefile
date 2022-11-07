#Makefile

.PHONY: default test clean

default:
	cd src; $(MAKE)

clean:
	cd src/; $(MAKE) clean
	cd tests/; $(MAKE) clean

test:
	cd tests; $(MAKE)
