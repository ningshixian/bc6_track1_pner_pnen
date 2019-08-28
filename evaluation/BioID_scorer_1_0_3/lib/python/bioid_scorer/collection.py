# Copyright 2017 The MITRE Corporation. All rights reserved.
# Approved for Public Release; Distribution Unlimited. Case Number 17-2967.

import os, glob, sys

from bioid_scorer.error import BioIDScoreError

from bioc import BioCReader

PYBIOC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PyBioC-1.0.2")

BIOC_DTD = os.path.join(PYBIOC_ROOT, "BioC.dtd")

class Collection:

    def __init__(self, d, verbose = 0):
        self.verbose = verbose
        self.directory = d
        self._load(d)

    def _load(self, d):
        self.biocDocs = []
        if self.verbose:
            print >> sys.stderr, "Reading", d
        filesRead = 0
        for xmlFile in glob.glob(os.path.join(d, "*.xml")):
            try:
                r = BioCReader(xmlFile, dtd_valid_file=BIOC_DTD)
                r.read()
            except Exception, e:
                raise BioIDScoreError, ("encountered error reading collection: %s" % str(e))
            # The collection contains a document for each caption.
            # Each caption contains a single passage.
            self.biocDocs.append(r.collection)
            filesRead += 1
            if self.verbose > 1:
                print >> sys.stderr, "Read", xmlFile
            elif self.verbose and (filesRead % 20 == 0):
                print >> sys.stderr, "...", filesRead
        if self.verbose:
            print >> sys.stderr, "...done."
        self._inferTypes()

    # The agreement in BioID is to score the following:
    # Entrez/Uniprot (genes and gene products)
    # Chebi (small chemicals - primary)
    # GoCC (subcellular)
    # Cellosaurus (cell lines)
    # Cell ontology (cell types)
    # Uberon (tissue)
    # NCBI taxon (organism)
    # Waiting on further instructions.
    
    COLON_PREFIX_MAP = {"Uniprot": ("gene_or_protein", True),
                        "protein": ("gene_or_protein", False),
                        "Corum": ("gene_or_protein", True), # actually protein complexes - won't be scored
                        "NCBI gene": ("gene_or_protein", True), # Also known as Entrez
                        "gene": ("gene_or_protein", False),
                        "Rfam": ("mRNA", True),
                        "mRNA": ("mRNA", False), # this isn't attested, but I use it in my sample output
                        "CHEBI": ("small_molecule", True),
                        "PubChem": ("small_molecule", True),
                        "molecule": ("small_molecule", False),
                        "BAO": ("small_molecule", True), # actually assays - won't be scored
                        "GO": ("cellular_component", True),
                        "subcellular": ("cellular_component", False),
                        "CL": ("cell_type_or_line", True), # Cell Ontology for cell lines
                        "cell": ("cell_type_or_line", False),
                        "Uberon": ("tissue_or_organ", True),
                        "tissue": ("tissue_or_organ", False),
                        "NCBI taxon": ("organism_or_species", True),
                        "organism": ("organism_or_species", False)}

    UNDERSCORE_PREFIX_MAP = {"CVCL": ("cell_type_or_line", True)} # Cellosaurus cell lines
    
    def _inferTypes(self):
        totals = dict([(k, 0) for k in self.COLON_PREFIX_MAP.keys() + self.UNDERSCORE_PREFIX_MAP.keys()])
        for c in self.biocDocs:
            for document in c.documents:
                for annotation in document.passages[0].annotations:
                    lab, grounded = None, False
                    # In the gold, the type may contain multiple
                    # grounds of the same type, separated by "|".
                    # This separator trick should still work.
                    tpe = annotation.infons["type"]
                    if ":" in tpe:
                        pref = tpe.split(":", 1)[0]
                        try:
                            lab, grounded = self.COLON_PREFIX_MAP[pref]
                        except KeyError:
                            pass
                    if (lab is None) and ("_" in tpe):
                        pref = tpe.split("_", 1)[0]
                        try:
                            lab, grounded = self.UNDERSCORE_PREFIX_MAP[pref]
                        except KeyError:
                            pass
                    # Last try: what if they used the generic
                    # name but no fake grounding?
                    if lab is None:
                        try:
                            lab, grounded = self.COLON_PREFIX_MAP[tpe]
                            if grounded:
                                lab = None
                            else:
                                pref = tpe
                        except KeyError:
                            pass
                    annotation._label = lab
                    annotation._grounded = (grounded and tpe) or None
                    if lab is not None:
                        totals[pref] += 1
                    else:
                        print >> sys.stderr, "No mapping for", tpe, "in", document.id
        if self.verbose:
            print "Type count:", totals
