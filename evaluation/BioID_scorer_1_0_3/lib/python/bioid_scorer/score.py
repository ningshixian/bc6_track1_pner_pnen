# Copyright 2017 The MITRE Corporation. All rights reserved.
# Approved for Public Release; Distribution Unlimited. Case Number 17-2967.

from bioid_scorer.error import BioIDScoreError
from bioid_scorer.collection import Collection

# We're going to aggregate these scores in multiple ways.
# We're going to have a single pairing; we're going to consider the label,
# and this may introduce a bit of perturbation in the label-agnositc
# case, but not much, because in the case of span clashes,
# only a vanishingly small number of cases have label 
# clashes as well. So we're going to ignore those; they'll
# be in the noise, and I can't be bothered with generating
# multiple pairings.

# We're going to do two pairings, though: one among the
# grounded annotations on both side, and one among all the annotations.
# THOSE might realistically differ.

# The documents are small enough that I'm not going to bother
# subdividing the regions of overlap. munkres will run OK.

import sys, os, csv, re
from munkres import Munkres, make_cost_matrix

EQ_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "resources", "equivalence_classes.json")

class PairSet:

    def __init__(self, pmcId, fig, refs, hyps):
        self.pmcId = pmcId
        self.figure = fig
        self.refAnnots = refs
        self.hypAnnots = hyps
        self._pairAnnotations()

    def _pairAnnotations(self):

        def annInfo(annot):
            # Sort the locations.
            tuples = [(int(l.offset), int(l.offset) + int(l.length)) for l in annot.locations]            
            firstStart = min([p[0] for p in tuples])
            lastEnd = max([p[1] for p in tuples])
            return firstStart, lastEnd, annot._label
        
        # Similarity is mostly overlap, a bit label.        
        def computeSimilarity(annotAtuple, annotBtuple, cache):
            a, (aLeft, aRight, aLabel) = annotAtuple
            b, (bLeft, bRight, bLabel) = annotBtuple
            try:
                return cache[(a, b)][0]
            except KeyError:
                pass
            overlap = (min(aRight, bRight) - max(aLeft, bLeft))/float(max(aRight, bRight) - min(aLeft, bLeft))
            clashes = []
            # If they don't overlap, they're not a pair, period.
            if overlap <= 0:
                cache[(a, b)] = (0, [])
                return 0
            if overlap < 1.0:
                clashes.append("spanclash")
            if aLabel != bLabel:
                clashes.append("labelclash")
            if not clashes:
                sim = 1.0
            elif aLabel == bLabel:
                sim = .1 + (.9 * overlap)
            else:
                sim = .9 * overlap
            cache[(a, b)] = (sim, clashes)
            return sim

        refAnnots = [(a, annInfo(a)) for a in self.refAnnots]
        hypAnnots = [(a, annInfo(a)) for a in self.hypAnnots]
        # Now, we have to compute the similarities and use munkres to align them.
        # This is taken pretty much exactly from the MAT scorer.
        missing = []
        spurious = []
        paired = []
        simCache = {}
        if refAnnots and hypAnnots:
            if (len(refAnnots) == 1) and (len(hypAnnots) == 1):
                # If they overlap, they're paired. Otherwise, no.
                sim = computeSimilarity(refAnnots[0], hypAnnots[0], simCache)
                if sim > 0:
                    paired = [(refAnnots[0], hypAnnots[0])]
                else:
                    spurious = hypAnnots
                    missing = refAnnots
            else:
                # Here is where we need munkres.
                matrix = make_cost_matrix([[computeSimilarity(r, h, simCache) for h in hypAnnots] for r in refAnnots],
                                          lambda cost: 1.0 - cost)
                indexPairs = Munkres().compute(matrix)
                for row, column in indexPairs:
                    try:
                        rAnn = refAnnots[row]
                    except IndexError:
                        # hyp matched with nothing.
                        spurious.append(hypAnnots[column])
                        continue
                    try:
                        hAnn = hypAnnots[column]
                    except IndexError:
                        # ref matched with nothing.
                        missing.append(refAnnots[row])
                        continue
                    # So now, what's their similarity?
                    if simCache[(rAnn[0], hAnn[0])][0] == 0:
                        spurious.append(hAnn)
                        missing.append(rAnn)
                    else:
                        paired.append((rAnn, hAnn))
                # Some of the elements aren't paired. They need to be in missing
                # or spurious.
                if len(indexPairs) < max(len(refAnnots), len(hypAnnots)):
                    allRef = set(missing) | set([r for (r, h) in paired])
                    allHyp = set(spurious) | set([h for (r, h) in paired])
                    missing += list(set(refAnnots) - allRef)
                    spurious += list(set(hypAnnots) - allHyp)
        elif refAnnots:
            missing = refAnnots
        elif hypAnnots:        
            spurious = hypAnnots
        self.pairs = [(p1[0], p2[0], simCache[(p1[0], p2[0])]) for (p1, p2) in paired]
        self.missing = [m[0] for m in missing]
        self.spurious = [s[0] for s in spurious]

# The aggregator for the data. We compute the conditions
# on the way OUT. Right now, we just aggregate.

# Originally, we were computing overall scores, but since the performers might
# intentionally do only some types of entities, aggregating overall scores
# isn't the right thing to do. This is ALSO true when computing the
# normalization scores, which should also be per label.

class Aggregator:

    def __init__(self, scorer, pmcId, fig):
        self.scorer = scorer
        self.pmcId = pmcId
        self.figure = fig
        self.labHash = {}
        self.parent = None
        self.captionCount = 0
    
    # Suck up the pair set, by label. Later, we aggregate when
    # we generate the spreadsheet. We keep track of the clashes
    # rather than counting them so that later, we can 
    # compute the conditions live.

    # We also want to compute the missing/spurious/match
    # for the grounded elements. This computation is complicated
    # by the fact that the gold might have alternative groundings.

    def incorporatePairSet(self, pairSet):
        self.captionCount = 1
        for m in pairSet.missing:
            try:
                refEntry = self.labHash[m._label]
                try:
                    refEntry["missing"] += 1
                except KeyError:
                    refEntry["missing"] = 1
            except KeyError:
                refEntry = {"missing": 1}
                self.labHash[m._label] = refEntry            
            if m._grounded:
                # I'm just going to collect them here, because
                # there's a bunch of complicated stuff that needs to
                # happen later.
                try:
                    refEntry["refgrounded"].append(m._grounded)
                except KeyError:
                    refEntry["refgrounded"] = [m._grounded]
        for s in pairSet.spurious:
            try:
                hypEntry = self.labHash[s._label]
                try:
                    hypEntry["spurious"] += 1
                except KeyError:
                    hypEntry["spurious"] = 1
            except KeyError:
                hypEntry = {"spurious": 1}
                self.labHash[s._label] = hypEntry
            if s._grounded:
                try:
                    hypEntry["hypgrounded"].append(s._grounded)
                except KeyError:
                    hypEntry["hypgrounded"] = [s._grounded]
        for (ref, hyp, (sim, clashes)) in pairSet.pairs:
            try:
                refEntry = self.labHash[ref._label]
            except KeyError:
                refEntry = {}
                self.labHash[ref._label] = refEntry                     
            if ref._grounded:
                try:
                    refEntry["refgrounded"].append(ref._grounded)
                except KeyError:
                    refEntry["refgrounded"] = [ref._grounded]
            try:
                hypEntry = self.labHash[hyp._label]
            except KeyError:
                hypEntry = {}
                self.labHash[hyp._label] = hypEntry
            if hyp._grounded:
                try:
                    hypEntry["hypgrounded"].append(hyp._grounded)
                except KeyError:
                    hypEntry["hypgrounded"] = [hyp._grounded]
            if sim == 1.0:
                try:
                    refEntry["match"] += 1
                except KeyError:
                    refEntry["match"] = 1
            else:
                try:
                    refEntry["refclash"].append(clashes)
                except KeyError:
                    refEntry["refclash"] = [clashes]
                try:
                    hypEntry["hypclash"].append(clashes)
                except KeyError:
                    hypEntry["hypclash"] = [clashes]

        # OK, we have hyp and ref grounded. Loop through
        # all the keys and compute.
        for v in self.labHash.values():
            hypGrounded = set([self.scorer._getGroundingID(gId) for gId in v.get("hypgrounded", [])])
            refGroundedSingletons = set()
            refGroundedSets = set()
            refGrounded = set()
            for gId in v.get("refgrounded", []):
                gIdSet = set([self.scorer._getGroundingID(gElt) for gElt in gId.split("|")])
                refGrounded |= gIdSet
                if len(gIdSet) == 1:
                    refGroundedSingletons |= gIdSet
                else:
                    refGroundedSets.add(frozenset(gIdSet))

            # Step 1: the matches are the intersection of hypGrounded and refGrounded.
            gMatchSet = hypGrounded & refGrounded

            # Step 2: the spurious are the hypGrounded minus the matches.
            gSpuriousSet = hypGrounded - gMatchSet

            # Step 3: remove any grounded set which intersects with the match set.
            # These have been satisfied.
            refGroundedSets = [s for s in refGroundedSets if not (s & gMatchSet)]

            # Step 4: compute the initial missing set.
            gMissingSet = refGroundedSingletons - gMatchSet

            # Step 5: remove the grounded sets which intersect with the missing set.
            # These, too, have been counted.
            refGroundedSets = [s for s in refGroundedSets if not (s & gMissingSet)]

            # Now, what we have left is those gold alternative sets which have no members
            # in the hypothesis IDs and have not been otherwise already documented as missing.
            # So we add to the missing count the lesser of (a) the number of remaining gold alternative
            # sets and (b) the size of the union of these sets.
            if refGroundedSets:
                missingBump = min(len(refGroundedSets), len(reduce(lambda x, y: x | y, refGroundedSets)))
            else:
                missingBump = 0

            gMatch = len(gMatchSet)
            gSpurious = len(gSpuriousSet)
            gMissing = len(gMissingSet) + missingBump

            # In some cases, there will be none of match, missing, spurious, because, e.g.,
            # the label has no grounded elements anywhere. In that case, we only accumulate the
            # macro scores if there's something there.
            
            v["gmatch"], v["gmissing"], v["gspurious"] = gMatch, gMissing, gSpurious
            gPrecision, gRecall, gFmeasure = self._prf(gMatch, 0, 0, gMissing, gSpurious)
            if gMatch + gMissing + gSpurious > 0:
                v["groundedRecalls"] = [gRecall]
                v["groundedPrecisions"] = [gPrecision]
                v["groundedFmeasures"] = [gFmeasure]
            else:
                v["groundedRecalls"] = []
                v["groundedPrecisions"] = []
                v["groundedFmeasures"] = []
            try:
                del v["hypgrounded"]
            except KeyError:
                pass
            try:
                del v["refgrounded"]
            except KeyError:
                pass

    def _prf(self, match, refClash, hypClash, missing, spurious):
        refOnly = refClash + missing
        refTotal = refOnly + match
        hypOnly = hypClash + spurious
        hypTotal = hypOnly + match
        if hypTotal == 0:
            precision = 1.0
        else:
            precision = match/float(hypTotal)
        if refTotal == 0:
            recall = 1.0
        else:
            recall = match/float(refTotal)
        if (precision == 0) and (recall == 0):
            fMeasure = 0.0
        else:
            fMeasure = 2 * ((precision * recall) / float(precision + recall))
        return precision, recall, fMeasure        
                
    def pushToParent(self):
        if not self.parent:
            return
        parent = self.parent
        parent.captionCount += self.captionCount
        for lab, hsh in self.labHash.items():
            try:
                thisHash = parent.labHash[lab]
                for k, v in hsh.items():
                    # clash and grounding are the non-numerics.
                    if k in ("refclash", "hypclash", "groundedRecalls", "groundedPrecisions", "groundedFmeasures"):
                        try:
                            thisHash[k] = thisHash[k] + v
                        except KeyError:
                            thisHash[k] = v[:]
                    else:
                        try:
                            thisHash[k] += v
                        except KeyError:
                            thisHash[k] = v
            except KeyError:
                parent.labHash[lab] = hsh.copy()

    # span condition is either strict or overlap
    # label condition is either sensitive or agnostic. NO! we're not
    # doing sensitive vs. agnostic anymore, because it's
    # not suitable to aggregate the labels. I'm going to leave
    # the code in, for someday, maybe.
    # They involve how the clashes are allocated.
    
    def getScoreLines(self, sCond, lCond):
        # First, I alphabetize the labels.
        # Collect the totals. Not using them right now.
        tMatch = tRefclash = tMissing = tHypclash = tSpurious = 0
        labs = sorted(self.labHash.keys())
        for lab in labs:
            entry = self.labHash[lab]
            missing, spurious, match = entry.get("missing", 0), entry.get("spurious", 0), entry.get("match", 0)
            refClashList, hypClashList = entry.get("refclash", []), entry.get("hypclash", [])
            if sCond == "strict":
                if lCond == "sensitive":
                    # Any clash will do.
                    refClash, hypClash = len(refClashList), len(hypClashList)
                else:
                    # The clashes are the ones with "spanclash" in it.
                    refClash = len([c for c in refClashList if "spanclash" in c])
                    hypClash = len([c for c in hypClashList if "spanclash" in c])
                    match += len(refClashList) - refClash
            else:
                if lCond == "sensitive":
                    # The clashes are the ones with "labelclash" in it.
                    refClash = len([c for c in refClashList if "labelclash" in c])
                    hypClash = len([c for c in hypClashList if "labelclash" in c])
                    match += len(refClashList) - refClash
                else:
                    # Everything's sloppy. They all count as matches.
                    refClash = hypClash = 0
                    match += len(refClashList)
            tMatch += match
            tRefclash += refClash
            tMissing += missing
            tHypclash += hypClash
            tSpurious += spurious
            l = self._computeLine(lab, match, refClash, missing, hypClash, spurious)
            gmicro = self._prf(entry["gmatch"], 0, 0, entry["gmissing"], entry["gspurious"])
            if not entry["groundedRecalls"]:
                macroRecall = None
            else:
                macroRecall = sum(entry["groundedRecalls"])/ float(len(entry["groundedRecalls"]))
            if not entry["groundedPrecisions"]:
                macroPrecision = None
            else:
                macroPrecision = sum(entry["groundedPrecisions"])/ float(len(entry["groundedPrecisions"]))
            if (not entry["groundedFmeasures"]):
                macroFmeasure = None
            else:
                macroFmeasure = sum(entry["groundedFmeasures"])/ float(len(entry["groundedFmeasures"]))
            yield l + [entry["gmatch"], entry["gmissing"], entry["gspurious"]] + list(gmicro) + [macroPrecision, macroRecall, macroFmeasure]
        # l = self._computeLine("<all>", tMatch, tRefclash, tMissing, tHypclash, tSpurious)
        # yield l + [None, None, None, None, None, None, None, None, None]

    # ["caption_count", "label", "match", "refclash", "missing", "refonly",
    # "reftotal", "hypclash", "spurious", "hyponly", "hyptotal", 
    # "precision", "recall", "fmeasure",
    # "norm_match", "norm_missing", "norm_spurious",
    # "norm_precision_micro", "norm_recall_micro", "norm_fmeasure_micro",
    # "norm_precision_macro", "norm_recall_macro", "norm_fmeasure_macro"]

    def _computeLine(self, lab, match, refClash, missing, hypClash, spurious):
        refOnly = refClash + missing
        refTotal = refOnly + match
        hypOnly = hypClash + spurious
        hypTotal = hypOnly + match
        precision, recall, fMeasure = self._prf(match, refClash, hypClash, missing, spurious)
        return [self.captionCount, lab, match, refClash, missing, refOnly,
                refTotal, hypClash, spurious, hypOnly, hypTotal,
                precision, recall, fMeasure]        

# So the idea is that for each run, we create a pair state for each caption,
# and then we aggregate the pairs through a number of layers and write
# out the CSV file.

class Scorer:
    
    def __init__(self, goldDir, runDirs, verbose = 0, testEquivalenceClasses = None, typeRestrictions = None):
        self.verbose = verbose
        self.typeRestrictions = typeRestrictions
        self.runResults = []
        self._digestEquivalenceClasses(otherClasses = testEquivalenceClasses)
        self._populateCollections(goldDir, runDirs)

    def _digestEquivalenceClasses(self, otherClasses = None):
        # Either use the otherClasses or EQ_JSON.
        import json
        fp = open(otherClasses or EQ_JSON, "r")
        l = json.loads(fp.read())
        fp.close()
        self._eqClassMap = {}
        for subL in l:
            lab = "|".join(sorted(subL))
            for e in subL:
                self._eqClassMap[e] = lab

    def _getGroundingID(self, gId):
        return self._eqClassMap.get(gId, gId)

    def _populateCollections(self, goldDir, runDirs):
        self.goldCollection = Collection(goldDir, verbose = self.verbose)
        self.goldCaptions = 0
        for doc in self.goldCollection.biocDocs:
            self.goldCaptions += len(doc.documents)
        self.runPairs = [(slug, Collection(d, verbose = self.verbose)) for (slug, d) in runDirs]

    def score(self, outDir):
        i = 1
        for (slug, run) in self.runPairs:
            if not slug:
                # slug = "run %d" % i
                slug = os.path.basename(run.directory)
                i += 1
            self.scoreRun(slug, run)
        self.writeSpreadsheets(outDir)

    # We're going to discard any annotations which are not in the list of
    # annotation types that the evaluation recognizes.

    GROUNDED_RE_EXCLUSION = re.compile("^(Corum|BAO):")
    
    def _discardIrrelevantAnnots(self, annotList):
        anns = [a for a in annotList if
                (a._label is not None) and (a._label != "mRNA") and ((a._grounded is None) or (self.GROUNDED_RE_EXCLUSION.match(a._grounded) is None))]
        if self.typeRestrictions is None:
            return anns
        else:
            return [a for a in anns if a._label in self.typeRestrictions]

    def scoreRun(self, slug, run):
        # The aggregator slots are aligned with how the spreadsheets
        # are going to be written, which is also why the corpusAggregators
        # is a list of one element.
        groundedCache = {"pairSets": [], "captionAggregators": [],
                         "documentAggregators": [], "corpusAggregators": [Aggregator(self, "<all>", "<all>")]}
        allCache = {"pairSets": [], "captionAggregators": [],
                    "documentAggregators": [], "corpusAggregators": [Aggregator(self, "<all>", "<all>")]}
        self.runResults.append((slug, run, allCache, groundedCache))
        # Create a hash of the run captions, since it's not necessarily
        # the case that every caption has annotations in each run.
        captionHash = {}
        for doc in run.biocDocs:
            for caption in doc.documents:
                captionHash[(caption.infons["pmc_id"], caption.infons["figure"])] = caption

        if self.verbose:
            print >> sys.stderr, "Scoring", self.goldCaptions, "gold captions against", slug, run.directory
        counts = 0
        for doc in self.goldCollection.biocDocs:
            for caption in doc.documents:
                refAnnots = self._discardIrrelevantAnnots(caption.passages[0].annotations)
                pmcId, fig = caption.infons["pmc_id"], caption.infons["figure"]
                hypAnnots = []
                try:
                    hypCaption = captionHash[(pmcId, fig)]
                    hypAnnots = self._discardIrrelevantAnnots(hypCaption.passages[0].annotations)
                except KeyError:
                    pass
                pairsAll = PairSet(pmcId, fig, refAnnots, hypAnnots)
                self.aggregatePair(pairsAll, allCache, pmcId, fig)
                pairsGrounded = PairSet(pmcId, fig, [a for a in refAnnots if a._grounded],
                                        [a for a in hypAnnots if a._grounded])
                self.aggregatePair(pairsGrounded, groundedCache, pmcId, fig)
                counts += 1
                if self.verbose > 1:
                    print >> sys.stderr, "  Scored", pmcId, fig
                elif self.verbose and (counts % 500 == 0):
                    print "...", counts
        for aggr in allCache["documentAggregators"]:
            aggr.pushToParent()
        for aggr in groundedCache["documentAggregators"]:
            aggr.pushToParent()
        if self.verbose:
            print "...done."

    def aggregatePair(self, pairSet, cache, pmcId, fig):
        cache["pairSets"].append(pairSet)
        # First, we introduce a new aggregator.
        pairAggregator = Aggregator(self, pmcId, fig)
        pairAggregator.incorporatePairSet(pairSet)
        cache["captionAggregators"].append(pairAggregator)
        # Now, we see whether the document just changed.
        # If it did, we introduce a new document aggregator.
        # Otherwise, we use the last one.
        if not cache["documentAggregators"]:
            docAggregator = Aggregator(self, pmcId, "<all>")
            cache["documentAggregators"].append(docAggregator)
        else:                
            docAggregator = cache["documentAggregators"][-1]
            if docAggregator.pmcId != pmcId:
                # This is the first aggregator for a new document.
                # Aggregate the document into the corpus aggregator
                # and move on.
                docAggregator = Aggregator(self, pmcId, "<all>")
                docAggregator.parent = cache["corpusAggregators"][0]
                cache["documentAggregators"].append(docAggregator)
        pairAggregator.parent = docAggregator
        pairAggregator.pushToParent()

    def writeSpreadsheets(self, outDir):
        # We have a list of 4-tuples in the run results:
        # (slug, collection, allCache, groundedCache).
        # Within each cache, all the aggregations have all been
        # percolated. All we need to do is write out the spreadsheets
        # in each condition.
        # The hierarchy goes like this: 
        # run -> grounded condition -> (strict/sloppy x labsensitive/agnostic)
        # We do this for the corpus level, the document level, and the caption level.
        self.writeMentionSpreadsheet("corpus", outDir, [])
        self.writeMentionSpreadsheet("document", outDir, [("document", "pmcId")])
        self.writeMentionSpreadsheet("caption", outDir, [("document", "pmcId"), ("figure", "figure")])
        self.writeDetailSpreadsheets(outDir)

    # We originally computed aggregates across labels, but it turns out that this is a terrible
    # idea because some performers will do some labels and not others. So no aggregation# (see
    # getScoreLines above), and no label conditions.

    def writeMentionSpreadsheet(self, level, outDir, prefixColumns):
        fp = open(os.path.join(outDir, level + "_scores.csv"), "w")
        w = csv.writer(fp)
        w.writerow(["run", "norm_condition", "span_condition"] + 
                   [p[0] for p in prefixColumns] +
                   ["caption_count", "label", "match", "refclash", "missing", "refonly",
                    "reftotal", "hypclash", "spurious", "hyponly", "hyptotal", 
                    "precision", "recall", "fmeasure",
                    "norm_match", "norm_missing", "norm_spurious",
                    "norm_precision_micro", "norm_recall_micro", "norm_fmeasure_micro",
                    "norm_precision_macro", "norm_recall_macro", "norm_fmeasure_macro"])
        for slug, collection, allCache, groundedCache in self.runResults:
            for (gCond, cache) in [("any", allCache), ("normalized", groundedCache)]:
                aggregators = cache[level+"Aggregators"]
                for sCond in ("strict", "overlap"):
                    for aggr in aggregators:
                        prefix = [slug, gCond, sCond] + \
                                 [getattr(aggr, p[1]).encode("utf-8") for p in prefixColumns]
                        for scoreLine in aggr.getScoreLines(sCond, "sensitive"):
                            w.writerow(prefix + scoreLine)
        fp.close()

    def writeDetailSpreadsheets(self, outDir):
        self._writeDetailSpreadsheet(outDir, "pair_details_no_norm_restriction.csv", 2)
        self._writeDetailSpreadsheet(outDir, "pair_details_normalized_only.csv", 3)

    def _writeDetailSpreadsheet(self, outDir, fName, idx):
        headers = ["run", "document", "figure", "status", 
                   "reflabel", "refstart", "refend", "reftext", "reftype", "reftype_eqclass",
                   "hyplabel", "hypstart", "hypend", "hyptext", "hyptype", "hyptype_eqclass"]
        # This is different: in this case, we do the two grounding conditions separately.
        fp = open(os.path.join(outDir, fName), "w")
        w = csv.writer(fp)
        w.writerow(headers)
        for res in self.runResults:
            slug = res[0]
            cache = res[idx]
            for pairSet in cache["pairSets"]:
                spurious = sorted(pairSet.spurious, key = lambda a: min([int(l.offset) for l in a.locations]))
                refRepresented = [(a, None, ["missing"]) for a in pairSet.missing] + \
                                 [(a, b, p[1]) for (a, b, p) in pairSet.pairs]
                refRepresented.sort(key = lambda p: min([int(l.offset) for l in p[0].locations]))
                for ref, hyp, errors in refRepresented:
                    if not errors:
                        status = "match"
                    else:
                        status = ",".join(errors)
                    
                    tuples = [(int(l.offset), int(l.offset) + int(l.length)) for l in ref.locations]
                    firstStart = min([p[0] for p in tuples])
                    lastEnd = max([p[1] for p in tuples])
                    hypLab =  hypStart = hypEnd = hypText = hypType = hypTypeEQ = None
                    # refGrounded has to be handled specially, since it might contain
                    # local alternatives.
                    refGroundedEQ = None
                    if ref._grounded:
                        cands = set([self._getGroundingID(g) for g in ref._grounded.split("|")])
                        if cands:
                            refGroundedEQ = "|".join(cands).encode("utf-8")                        
                    
                    if hyp:
                        hypLab = hyp._label
                        tuples = [(int(l.offset), int(l.offset) + int(l.length)) for l in hyp.locations]
                        hypStart = min([p[0] for p in tuples])
                        hypEnd = max([p[1] for p in tuples])
                        hypText = hyp.text.encode("utf-8")
                        hypType = (hyp._grounded and hyp._grounded.encode("utf-8")) or None
                        hypTypeEQ = (hyp._grounded and self._getGroundingID(hyp._grounded)) or None
                        
                    w.writerow([slug, pairSet.pmcId, pairSet.figure.encode("utf-8"), status,
                                ref._label, firstStart, lastEnd, ref.text.encode("utf-8"),
                                (ref._grounded and ref._grounded.encode("utf-8")) or None,
                                refGroundedEQ,
                                hypLab, hypStart, hypEnd, hypText, hypType, hypTypeEQ])                    
                    
                for spurious in pairSet.spurious:
                    tuples = [(int(l.offset), int(l.offset) + int(l.length)) for l in spurious.locations]
                    firstStart = min([p[0] for p in tuples])
                    lastEnd = max([p[1] for p in tuples])
                    w.writerow([slug, pairSet.pmcId, pairSet.figure.encode("utf-8"), "spurious",
                                None, None, None, None, None, None,
                                spurious._label, firstStart, lastEnd, spurious.text.encode("utf-8"),
                                (spurious._grounded and spurious._grounded.encode("utf-8")) or None,
                                (spurious._grounded and self._getGroundingID(spurious._grounded)) or None])

# Toplevel.

def Score(goldDir, runDirs, outCsv, verbose = 0, testEquivalenceClasses = None, typeRestrictions = None):
    Scorer(goldDir, runDirs, verbose = verbose, testEquivalenceClasses = testEquivalenceClasses, typeRestrictions = typeRestrictions).score(outCsv)
