# Code Adapted from Peter de Vocht's
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from quantulum3 import parser
import pandas as pd

# use spacy small model
nlp = en_core_web_sm.load()

# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}



# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet):
    return "and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ == "NOUN":
        return [head], _is_negated(tok)
    return [], False


# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False


# get all the verbs on tokens with negation marker
def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# xcomp; open complement - verb has no suject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET" and tok.lower_ not in STOP_WORDS]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


# is the token a verb?  (excluding auxiliary verbs)
def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass")


# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    #potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs


# return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(item, toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return item


# simple stemmer using lemmas
def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


# print information for displaying all kinds of things of the parse tree
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


# expand an obj / subj np using its chunk
def expand(item, tokens, visited):
    if item.lower_ == 'that':
        item = _get_that_resolution(item, tokens)

    parts = []
    node = get_node(item)
    if node:
        parts.append(node)

    return parts


# convert a list of tokens to a string
def to_str(tokens):
    return ' '.join([item for item in tokens])


# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens):
    svos = []
    is_pas = _is_passive(tokens)
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    visited = set()  # recursion detection

    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0 :
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expand(obj, tokens, visited)),
                                          to_str(expand(sub, tokens, visited)), v.lemma_ + ' not' if verbNegated or objNegated else v.lemma_))
                            svos.append((to_str(expand(obj, tokens, visited)),
                                           to_str(expand(sub, tokens, visited)), v2.lemma_ + ' not' if verbNegated or objNegated else v2.lemma_))
                        else:
                            svos.append((to_str(expand(sub, tokens, visited)),
                                          to_str(expand(obj, tokens, visited)), v.lemma_ + ' not' if verbNegated or objNegated else v.lemma_))
                            svos.append((to_str(expand(sub, tokens, visited)),
                                          to_str(expand(obj, tokens, visited)), v2.lemma_ + ' not' if verbNegated or objNegated else v2.lemma_))
            else:
                v, objs = _get_all_objs(v, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)

                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expand(obj, tokens, visited)),
                                          to_str(expand(sub, tokens, visited)), v.lemma_ + ' not' if verbNegated or objNegated else v.lemma_))
                        else:
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         to_str(expand(obj, tokens, visited)), v.lemma_ + ' not' if verbNegated or objNegated else v.lemma_))
    
    return [(s,o,v) for s,o,v in svos if (s!='') and (o!='')]

def get_text(file, sep):
    cont = list()
    with open(file, 'r') as file:
        line = file.readlines()
    for abstr_ in line:
        sentences = abstr_.strip().split(sep)
        cont.extend([sen_ for sen_ in sentences[1:]  if (sen_ != '')&(~sen_.isnumeric())])
    return cont

def extract_nodes(sentence):
    list_nodes = list()
    for word in sentence:
        if word.lower_ in STOP_WORDS:
            pass
        rights = list(word.rights)
        lefts = list(word.lefts)
        if word.pos_ == 'NOUN':
            for tok in lefts:
                if tok.dep_ == 'compound' or tok.dep_ == 'amod' or tok.pos_ == 'NUM': # to be refined
                    if tok.lower_ not in STOP_WORDS:
                        list_nodes.append(tok.lemma_+' '+word.lemma_)
                    else:
                        list_nodes.append(word.lemma_)
                else:
                    list_nodes.append(word.lemma_)
    return list_nodes

def get_node(tok):
    if tok.lower_ in STOP_WORDS:
        return None

    lefts = [x for x in list(tok.lefts) if x.lower_ not in STOP_WORDS]
    if len(lefts)>0:
        
        for left in lefts:

            if left.dep_ == 'compound' or left.dep_ == 'amod':# or ((tok.lower_ == '%') and (left.pos_ == 'NUM')): # to be refined
                return (left.lemma_+' '+tok.lemma_)
            
            if tok.pos_ == 'NOUN' and left.pos_ == 'NUM':
                quants = parser.parse(left.lower_ + tok.lower_)
                if quants and quants[0].unit.name != 'dimensionless':
                    return(left.lower_ + tok.lower_)
                
    
            else:
                return tok.lemma_
    else:
        return tok.lemma_

def extract_link(sen):
    list_edges = list()

    preps = [tok for tok in sen if tok.pos_ == "ADP" and tok.dep_ == "prep"]    # covers the cases where a verb has no subject?
    for prep in preps:
#         lefts = [tok for tok in prep.lefts if tok.pos_ == 'NOUN']
        left = prep.head
        rights = [tok for tok in prep.rights if tok.pos_ == 'NOUN']
        if len(rights)>0:
            left_node = get_node(left)
            right_node = get_node(rights[0])
            if (left_node is not None) and (right_node is not None):
                list_edges.append((left_node,right_node,prep.lemma_))

    list_edges.extend(findSVOs(sen))

    return list_edges

def get_sentences(curr_abstract):
    sentences = nltk.tokenize.sent_tokenize(curr_abstract)
    #cont = [sen_ for sen_ in sentences[1:]  if (sen_ != '')&(~sen_.isnumeric())]
    return sentences


def main():
    # Need to Change the path
    path_to_stopwords = '/Users/mac/Documents/CodeDirectory/nltk_stopwords.txt'
    
    # Keep this block
    with open(path_to_stopwords, 'r') as file:
        stopwords = file.readlines()
    for word in stopwords:
        nlp.vocab[word].is_stop = True

    # change this
    #all_text = get_text('/Users/mac/Documents/CodeDirectory/processed_abstracts_old.csv', ',')
    # all_text = 'the corona virus sars-cov-2 or covid-19 pandemic is growing alarmingly throughout the whole world. using the power law scaling we analyze the data of different countries and three states of india up to 1st april 2020 and explain in terms of power law exponent. we find significant reduction in growth of infections in china and denmark g reduced from approximately 2.18 to 0.05 and 11.41 to 6.95 respectively. very slow reduction is also seen in brazil and germany g reduced from approximately 6 to 4 and 11 to 7 respectively. infection in india is growing g9.23 though lesser in number than that in the usa highest g of 16 approximately studied so far italy and a few other countries'
    # links = extract_link(nlp(all_text))
    # print(links)

    data = pd.read_csv('/Users/mac/Documents/CodeDirectory/processed_dataset.csv')
    data['links'] = ''

    for ix, row in data.iterrows():
        print(ix)
        data.loc[ix, 'links'] = extract_link(nlp(row['abstract']))

    data.to_csv('/Users/mac/Documents/CodeDirectory/test_extract.csv')

if __name__ == '__main__':
    main()







