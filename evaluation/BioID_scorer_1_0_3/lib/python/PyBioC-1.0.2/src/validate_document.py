#!/usr/bin/env python

import os, sys

def Usage():
    print >> sys.stderr, "Usage: validate_document.py input_doc..."
    sys.exit(1)

if len(sys.argv) < 2:
    Usage()

THIS_ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_ROOT)

INPUTS = sys.argv[1:]

sys.path.insert(0, THIS_ROOT)

from bioc import BioCReader
from bioc import BioCWriter

dtd_file = os.path.join(PKG_ROOT, "BioC.dtd")
print "DTD is", dtd_file

validated = 0

for test_file in INPUTS:
    try:
        bioc_reader = BioCReader(test_file, dtd_valid_file=dtd_file)
        bioc_reader.read()
        validated += 1
    except Exception, e:
        print >> sys.stderr, "For", test_file, ":", str(e)


print validated, "of", len(INPUTS), "validated."
