"""CLAWS7 tag → high-level POS and attribute mapping.

Two public objects:
    CLAWS7_POS         : dict[str, str]   base_tag → POS label
    CLAWS7_ATTRIBUTES  : dict[str, dict]  base_tag → subclass attribute dict

Covers all tags present in data/snli_CLAWS7.txt (plus the full CLAWS7 tagset
for completeness). Ditto tag stripping (II31 → II) is handled by the caller.

Reference: https://ucrel.lancs.ac.uk/claws7tags.html
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# High-level POS labels
# ---------------------------------------------------------------------------

CLAWS7_POS: dict[str, str] = {
    # --- NOUN ----------------------------------------------------------------
    "NN":    "NOUN",   # common noun, number-neutral (sheep, cod)
    "NN1":   "NOUN",   # singular common noun
    "NN2":   "NOUN",   # plural common noun
    "NNA":   "NOUN",   # following noun of title (M.A.)
    "NNB":   "NOUN",   # preceding noun of title (Mr., Prof.)
    "NNL1":  "NOUN",   # singular locative noun (Street, Island)
    "NNL2":  "NOUN",   # plural locative noun
    "NNO":   "NOUN",   # numeral noun, number-neutral (dozen, hundred)
    "NNO2":  "NOUN",   # numeral noun, plural (hundreds, thousands)
    "NNT1":  "NOUN",   # temporal noun, singular (day, week)
    "NNT2":  "NOUN",   # temporal noun, plural
    "NNU":   "NOUN",   # unit of measurement, number-neutral (in, cc)
    "NNU1":  "NOUN",   # unit, singular (inch)
    "NNU2":  "NOUN",   # unit, plural (feet)
    "NP":    "NOUN",   # proper noun, number-neutral (IBM, Andes)
    "NP1":   "NOUN",   # singular proper noun
    "NP2":   "NOUN",   # plural proper noun
    "NPD1":  "NOUN",   # weekday noun, singular (Sunday)
    "NPD2":  "NOUN",   # weekday noun, plural
    "NPM1":  "NOUN",   # month noun, singular (October)
    "NPM2":  "NOUN",   # month noun, plural
    "ND1":   "NOUN",   # direction noun (north, southeast)

    # --- VERB ----------------------------------------------------------------
    # be-forms
    "VB0":   "VERB",   # be, base form (finite: imperative, subjunctive)
    "VBDR":  "VERB",   # were
    "VBDZ":  "VERB",   # was
    "VBG":   "VERB",   # being
    "VBI":   "VERB",   # be, infinitive
    "VBM":   "VERB",   # am
    "VBN":   "VERB",   # been
    "VBR":   "VERB",   # are
    "VBZ":   "VERB",   # is
    # do-forms
    "VD0":   "VERB",   # do, base form (finite)
    "VDD":   "VERB",   # did
    "VDG":   "VERB",   # doing
    "VDI":   "VERB",   # do, infinitive
    "VDN":   "VERB",   # done
    "VDZ":   "VERB",   # does
    # have-forms
    "VH0":   "VERB",   # have, base form (finite)
    "VHD":   "VERB",   # had (past tense)
    "VHG":   "VERB",   # having
    "VHI":   "VERB",   # have, infinitive
    "VHN":   "VERB",   # had (past participle)
    "VHZ":   "VERB",   # has
    # modals
    "VM":    "VERB",   # modal auxiliary (can, will, would…)
    "VMK":   "VERB",   # modal catenative (ought, used)
    # lexical verbs
    "VV0":   "VERB",   # base form (give, work)
    "VVD":   "VERB",   # past tense (gave, worked)
    "VVG":   "VERB",   # -ing participle (giving, working)
    "VVGK":  "VERB",   # -ing participle catenative (going in be going to)
    "VVI":   "VERB",   # infinitive (to give, will work)
    "VVN":   "VERB",   # past participle (given, worked)
    "VVNK":  "VERB",   # past participle catenative (bound in be bound to)
    "VVZ":   "VERB",   # -s form (gives, works)

    # --- ADJECTIVE -----------------------------------------------------------
    "JJ":    "ADJ",    # general adjective
    "JJR":   "ADJ",    # comparative adjective (older, better)
    "JJT":   "ADJ",    # superlative adjective (oldest, best)
    "JK":    "ADJ",    # catenative adjective (able in be able to)

    # --- ADVERB --------------------------------------------------------------
    "RR":    "ADV",    # general adverb
    "RRR":   "ADV",    # comparative general adverb (better, longer)
    "RRT":   "ADV",    # superlative general adverb (best, longest)
    "RRQ":   "ADV",    # wh- general adverb (where, when, why, how)
    "RRQV":  "ADV",    # wh-ever general adverb (wherever, whenever)
    "RG":    "ADV",    # degree adverb (very, so, too)
    "RGR":   "ADV",    # comparative degree adverb (more, less)
    "RGT":   "ADV",    # superlative degree adverb (most, least)
    "RGQ":   "ADV",    # wh- degree adverb (how)
    "RGQV":  "ADV",    # wh-ever degree adverb (however)
    "RL":    "ADV",    # locative adverb (alongside, forward)
    "RA":    "ADV",    # adverb after nominal head (else, galore)
    "REX":   "ADV",    # adverb introducing appositional constructions (namely)
    "RT":    "ADV",    # quasi-nominal adverb of time (now, tomorrow)

    # --- PREPOSITION ---------------------------------------------------------
    "II":    "PREP",   # general preposition
    "IF":    "PREP",   # for (as preposition)
    "IO":    "PREP",   # of (as preposition)
    "IW":    "PREP",   # with / without

    # --- DETERMINER ----------------------------------------------------------
    "AT":    "DET",    # article, number-neutral (the, no)
    "AT1":   "DET",    # singular article (a, an, every)
    "DD":    "DET",    # determiner, number-neutral (any, some)
    "DD1":   "DET",    # singular determiner (this, that, another)
    "DD2":   "DET",    # plural determiner (these, those)
    "DDQ":   "DET",    # wh-determiner (which, what)
    "DDQGE": "DET",    # wh-determiner, genitive (whose)
    "DDQV":  "DET",    # wh-ever determiner (whichever, whatever)
    "DA":    "DET",    # after-/post-determiner, number-neutral (such, same)
    "DA1":   "DET",    # singular post-determiner (little, much)
    "DA2":   "DET",    # plural post-determiner (few, several, many)
    "DAR":   "DET",    # comparative post-determiner (more, less, fewer)
    "DAT":   "DET",    # superlative post-determiner (most, least)
    "DB":    "DET",    # pre-determiner, number-neutral (all, half)
    "DB2":   "DET",    # plural pre-determiner (both)
    # Possessive pronoun used attributively (my, your, his, her, its, our, their)
    "APPGE": "DET",

    # --- PRONOUN -------------------------------------------------------------
    "PPGE":  "PRON",   # nominal possessive personal pronoun (mine, yours)
    "PPH1":  "PRON",   # 3rd person sing. neuter (it)
    "PPHO1": "PRON",   # 3rd person sing. objective (him, her)
    "PPHO2": "PRON",   # 3rd person pl. objective (them)
    "PPHS1": "PRON",   # 3rd person sing. subjective (he, she)
    "PPHS2": "PRON",   # 3rd person pl. subjective (they)
    "PPIO1": "PRON",   # 1st person sing. objective (me)
    "PPIO2": "PRON",   # 1st person pl. objective (us)
    "PPIS1": "PRON",   # 1st person sing. subjective (I)
    "PPIS2": "PRON",   # 1st person pl. subjective (we)
    "PPX1":  "PRON",   # singular reflexive (yourself, itself)
    "PPX2":  "PRON",   # plural reflexive (yourselves, themselves)
    "PPY":   "PRON",   # 2nd person (you)
    "PN":    "PRON",   # indefinite pronoun, number-neutral (none)
    "PN1":   "PRON",   # indefinite pronoun, singular (anyone, nobody, one)
    "PNQO":  "PRON",   # objective wh-pronoun (whom)
    "PNQS":  "PRON",   # subjective wh-pronoun (who)
    "PNQV":  "PRON",   # wh-ever pronoun (whoever)
    "PNX1":  "PRON",   # reflexive indefinite pronoun (oneself)

    # --- CONJUNCTION ---------------------------------------------------------
    "CC":    "CONJ",   # coordinating conjunction (and, or)
    "CCB":   "CONJ",   # adversative coordinating conjunction (but)
    "CS":    "CONJ",   # subordinating conjunction (if, because, unless)
    "CSA":   "CONJ",   # as (conjunction)
    "CSN":   "CONJ",   # than (conjunction)
    "CST":   "CONJ",   # that (conjunction)
    "CSW":   "CONJ",   # whether (conjunction)
    "BCL":   "CONJ",   # before-clause marker (in order that / to)

    # --- PARTICLE / INFINITIVE MARKER ----------------------------------------
    "RP":    "PART",   # prepositional adverb / particle (about, in, out, up)
    "RPK":   "PART",   # catenative particle (about in be about to)
    "TO":    "PART",   # infinitive marker (to)

    # --- NUMBER --------------------------------------------------------------
    "MC":    "NUM",    # cardinal number, number-neutral (two, three)
    "MC1":   "NUM",    # singular cardinal (one)
    "MC2":   "NUM",    # plural cardinal (sixes, sevens)
    "MCGE":  "NUM",    # genitive cardinal (two's, 100's)
    "MCMC":  "NUM",    # hyphenated number (40-50)
    "MD":    "NUM",    # ordinal number (first, second, next, last)
    "MF":    "NUM",    # fraction (quarters, two-thirds)

    # --- INTERJECTION --------------------------------------------------------
    "UH":    "INTJ",

    # --- SPECIAL CLASSES -----------------------------------------------------
    "EX":    "EXPL",   # existential there
    "XX":    "NEG",    # not, n't
    "GE":    "GEN",    # germanic genitive marker (' or 's)

    # --- UNCLASSIFIED / FOREIGN ----------------------------------------------
    "FW":    "X",
    "FO":    "X",
    "FU":    "X",
    "ZZ1":   "X",
    "ZZ2":   "X",
}


# ---------------------------------------------------------------------------
# Subclass attribute dicts
# ---------------------------------------------------------------------------

CLAWS7_ATTRIBUTES: dict[str, dict[str, str]] = {

    # ── NOUNS ──────────────────────────────────────────────────────────────
    "NN":   {"proper_common": "common", "number": "neutral"},
    "NN1":  {"proper_common": "common", "number": "singular"},
    "NN2":  {"proper_common": "common", "number": "plural"},
    "NNA":  {"proper_common": "common", "noun_subtype": "title_following"},
    "NNB":  {"proper_common": "common", "noun_subtype": "title_preceding"},
    "NNL1": {"proper_common": "common", "number": "singular", "noun_subtype": "locative"},
    "NNL2": {"proper_common": "common", "number": "plural",   "noun_subtype": "locative"},
    "NNO":  {"proper_common": "common", "number": "neutral",  "noun_subtype": "numeral"},
    "NNO2": {"proper_common": "common", "number": "plural",   "noun_subtype": "numeral"},
    "NNT1": {"proper_common": "common", "number": "singular", "noun_subtype": "temporal"},
    "NNT2": {"proper_common": "common", "number": "plural",   "noun_subtype": "temporal"},
    "NNU":  {"proper_common": "common", "number": "neutral",  "noun_subtype": "measure_unit"},
    "NNU1": {"proper_common": "common", "number": "singular", "noun_subtype": "measure_unit"},
    "NNU2": {"proper_common": "common", "number": "plural",   "noun_subtype": "measure_unit"},
    "NP":   {"proper_common": "proper", "number": "neutral"},
    "NP1":  {"proper_common": "proper", "number": "singular"},
    "NP2":  {"proper_common": "proper", "number": "plural"},
    "NPD1": {"proper_common": "proper", "number": "singular", "noun_subtype": "weekday"},
    "NPD2": {"proper_common": "proper", "number": "plural",   "noun_subtype": "weekday"},
    "NPM1": {"proper_common": "proper", "number": "singular", "noun_subtype": "month"},
    "NPM2": {"proper_common": "proper", "number": "plural",   "noun_subtype": "month"},
    "ND1":  {"proper_common": "common", "number": "singular", "noun_subtype": "directional"},

    # ── VERBS: be-forms ────────────────────────────────────────────────────
    "VB0":  {"verb_class": "copular", "lemma": "be", "finiteness": "finite",
             "verb_form": "base"},
    "VBDR": {"verb_class": "copular", "lemma": "be", "finiteness": "finite",
             "verb_form": "past", "number": "plural"},
    "VBDZ": {"verb_class": "copular", "lemma": "be", "finiteness": "finite",
             "verb_form": "past", "number": "singular"},
    "VBG":  {"verb_class": "copular", "lemma": "be", "finiteness": "non-finite",
             "verb_form": "present_participle"},
    "VBI":  {"verb_class": "copular", "lemma": "be", "finiteness": "non-finite",
             "verb_form": "infinitive"},
    "VBM":  {"verb_class": "copular", "lemma": "be", "finiteness": "finite",
             "verb_form": "present", "person": "1st", "number": "singular"},
    "VBN":  {"verb_class": "copular", "lemma": "be", "finiteness": "non-finite",
             "verb_form": "past_participle"},
    "VBR":  {"verb_class": "copular", "lemma": "be", "finiteness": "finite",
             "verb_form": "present", "number": "plural"},
    "VBZ":  {"verb_class": "copular", "lemma": "be", "finiteness": "finite",
             "verb_form": "present", "person": "3rd", "number": "singular"},

    # ── VERBS: do-forms ────────────────────────────────────────────────────
    "VD0":  {"verb_class": "auxiliary", "lemma": "do", "finiteness": "finite",
             "verb_form": "base"},
    "VDD":  {"verb_class": "auxiliary", "lemma": "do", "finiteness": "finite",
             "verb_form": "past"},
    "VDG":  {"verb_class": "auxiliary", "lemma": "do", "finiteness": "non-finite",
             "verb_form": "present_participle"},
    "VDI":  {"verb_class": "auxiliary", "lemma": "do", "finiteness": "non-finite",
             "verb_form": "infinitive"},
    "VDN":  {"verb_class": "auxiliary", "lemma": "do", "finiteness": "non-finite",
             "verb_form": "past_participle"},
    "VDZ":  {"verb_class": "auxiliary", "lemma": "do", "finiteness": "finite",
             "verb_form": "present", "person": "3rd", "number": "singular"},

    # ── VERBS: have-forms ──────────────────────────────────────────────────
    "VH0":  {"verb_class": "auxiliary", "lemma": "have", "finiteness": "finite",
             "verb_form": "base"},
    "VHD":  {"verb_class": "auxiliary", "lemma": "have", "finiteness": "finite",
             "verb_form": "past"},
    "VHG":  {"verb_class": "auxiliary", "lemma": "have", "finiteness": "non-finite",
             "verb_form": "present_participle"},
    "VHI":  {"verb_class": "auxiliary", "lemma": "have", "finiteness": "non-finite",
             "verb_form": "infinitive"},
    "VHN":  {"verb_class": "auxiliary", "lemma": "have", "finiteness": "non-finite",
             "verb_form": "past_participle"},
    "VHZ":  {"verb_class": "auxiliary", "lemma": "have", "finiteness": "finite",
             "verb_form": "present", "person": "3rd", "number": "singular"},

    # ── VERBS: modals ──────────────────────────────────────────────────────
    "VM":   {"verb_class": "modal",            "finiteness": "finite"},
    "VMK":  {"verb_class": "modal_catenative", "finiteness": "finite"},

    # ── VERBS: lexical ─────────────────────────────────────────────────────
    "VV0":  {"verb_class": "lexical",            "finiteness": "finite",
             "verb_form": "base"},
    "VVD":  {"verb_class": "lexical",            "finiteness": "finite",
             "verb_form": "past"},
    "VVG":  {"verb_class": "lexical",            "finiteness": "non-finite",
             "verb_form": "present_participle"},
    "VVGK": {"verb_class": "lexical_catenative", "finiteness": "non-finite",
             "verb_form": "present_participle"},
    "VVI":  {"verb_class": "lexical",            "finiteness": "non-finite",
             "verb_form": "infinitive"},
    "VVN":  {"verb_class": "lexical",            "finiteness": "non-finite",
             "verb_form": "past_participle"},
    "VVNK": {"verb_class": "lexical_catenative", "finiteness": "non-finite",
             "verb_form": "past_participle"},
    "VVZ":  {"verb_class": "lexical",            "finiteness": "finite",
             "verb_form": "third_singular_present"},

    # ── ADJECTIVES ─────────────────────────────────────────────────────────
    "JJ":   {"degree": "positive"},
    "JJR":  {"degree": "comparative"},
    "JJT":  {"degree": "superlative"},
    "JK":   {"adj_subtype": "catenative"},

    # ── ADVERBS ────────────────────────────────────────────────────────────
    "RR":   {"adverb_type": "general"},
    "RRR":  {"adverb_type": "general",  "degree": "comparative"},
    "RRT":  {"adverb_type": "general",  "degree": "superlative"},
    "RRQ":  {"adverb_type": "wh_general"},
    "RRQV": {"adverb_type": "wh_ever_general"},
    "RG":   {"adverb_type": "degree"},
    "RGR":  {"adverb_type": "degree",   "degree": "comparative"},
    "RGT":  {"adverb_type": "degree",   "degree": "superlative"},
    "RGQ":  {"adverb_type": "wh_degree"},
    "RGQV": {"adverb_type": "wh_ever_degree"},
    "RL":   {"adverb_type": "locative"},
    "RA":   {"adverb_type": "post_nominal"},
    "REX":  {"adverb_type": "appositional"},
    "RT":   {"adverb_type": "temporal"},

    # ── PREPOSITIONS ───────────────────────────────────────────────────────
    "II":   {"prep_subtype": "general"},
    "IF":   {"prep_subtype": "for"},
    "IO":   {"prep_subtype": "of"},
    "IW":   {"prep_subtype": "with_without"},

    # ── DETERMINERS ────────────────────────────────────────────────────────
    "AT":    {"det_type": "article",          "number": "neutral"},
    "AT1":   {"det_type": "article",          "number": "singular"},
    "DD":    {"det_type": "general",          "number": "neutral"},
    "DD1":   {"det_type": "demonstrative",    "number": "singular"},
    "DD2":   {"det_type": "demonstrative",    "number": "plural"},
    "DDQ":   {"det_type": "wh_interrogative"},
    "DDQGE": {"det_type": "wh_interrogative", "case": "genitive"},
    "DDQV":  {"det_type": "wh_ever"},
    "DA":    {"det_type": "post_determiner",  "number": "neutral"},
    "DA1":   {"det_type": "post_determiner",  "number": "singular"},
    "DA2":   {"det_type": "post_determiner",  "number": "plural"},
    "DAR":   {"det_type": "post_determiner",  "degree": "comparative"},
    "DAT":   {"det_type": "post_determiner",  "degree": "superlative"},
    "DB":    {"det_type": "pre_determiner",   "number": "neutral"},
    "DB2":   {"det_type": "pre_determiner",   "number": "plural"},
    "APPGE": {"det_type": "possessive_pronoun", "function": "attributive"},

    # ── PRONOUNS ───────────────────────────────────────────────────────────
    "PPGE":  {"pron_type": "possessive",                "function": "nominal"},
    "PPH1":  {"pron_type": "personal",  "person": "3rd", "number": "singular",
              "gender": "neuter"},
    "PPHO1": {"pron_type": "personal",  "person": "3rd", "number": "singular",
              "case": "objective"},
    "PPHO2": {"pron_type": "personal",  "person": "3rd", "number": "plural",
              "case": "objective"},
    "PPHS1": {"pron_type": "personal",  "person": "3rd", "number": "singular",
              "case": "subjective"},
    "PPHS2": {"pron_type": "personal",  "person": "3rd", "number": "plural",
              "case": "subjective"},
    "PPIO1": {"pron_type": "personal",  "person": "1st", "number": "singular",
              "case": "objective"},
    "PPIO2": {"pron_type": "personal",  "person": "1st", "number": "plural",
              "case": "objective"},
    "PPIS1": {"pron_type": "personal",  "person": "1st", "number": "singular",
              "case": "subjective"},
    "PPIS2": {"pron_type": "personal",  "person": "1st", "number": "plural",
              "case": "subjective"},
    "PPX1":  {"pron_type": "reflexive", "number": "singular"},
    "PPX2":  {"pron_type": "reflexive", "number": "plural"},
    "PPY":   {"pron_type": "personal",  "person": "2nd"},
    "PN":    {"pron_type": "indefinite", "number": "neutral"},
    "PN1":   {"pron_type": "indefinite", "number": "singular"},
    "PNQO":  {"pron_type": "wh_relative_interrogative", "case": "objective"},
    "PNQS":  {"pron_type": "wh_relative_interrogative", "case": "subjective"},
    "PNQV":  {"pron_type": "wh_ever"},
    "PNX1":  {"pron_type": "reflexive_indefinite"},

    # ── CONJUNCTIONS ───────────────────────────────────────────────────────
    "CC":    {"conj_type": "coordinating"},
    "CCB":   {"conj_type": "coordinating",  "conj_subtype": "adversative"},
    "CS":    {"conj_type": "subordinating"},
    "CSA":   {"conj_type": "subordinating", "lexeme": "as"},
    "CSN":   {"conj_type": "subordinating", "lexeme": "than"},
    "CST":   {"conj_type": "subordinating", "lexeme": "that"},
    "CSW":   {"conj_type": "subordinating", "lexeme": "whether"},
    "BCL":   {"conj_type": "before_clause_marker"},

    # ── PARTICLES / INFINITIVE MARKER ──────────────────────────────────────
    "RP":    {"particle_type": "prepositional"},
    "RPK":   {"particle_type": "catenative"},
    "TO":    {"particle_type": "infinitive_marker"},

    # ── NUMBERS ────────────────────────────────────────────────────────────
    "MC":    {"num_type": "cardinal", "number": "neutral"},
    "MC1":   {"num_type": "cardinal", "number": "singular"},
    "MC2":   {"num_type": "cardinal", "number": "plural"},
    "MCGE":  {"num_type": "cardinal", "case": "genitive"},
    "MCMC":  {"num_type": "cardinal", "num_subtype": "hyphenated"},
    "MD":    {"num_type": "ordinal"},
    "MF":    {"num_type": "fraction"},

    # ── SPECIAL ────────────────────────────────────────────────────────────
    "EX":    {"expletive_type": "existential"},
    "XX":    {},   # negation marker — no subclass attributes
    "GE":    {},   # germanic genitive marker

    # ── UNCLASSIFIED / FOREIGN ─────────────────────────────────────────────
    "FW":    {"word_type": "foreign"},
    "FO":    {"word_type": "formula"},
    "FU":    {"word_type": "unclassified"},
    "ZZ1":   {"word_type": "letter", "number": "singular"},
    "ZZ2":   {"word_type": "letter", "number": "plural"},

    # ── INTERJECTION ───────────────────────────────────────────────────────
    "UH":    {},
}
