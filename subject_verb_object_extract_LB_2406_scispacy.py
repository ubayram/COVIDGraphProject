# Code Adapted from Peter de Vocht's subject verb extract
#

## Instructions

# 1. download the large scispacy model for NER: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
# 2. download en_core_web_sm model for POS
# !python -m spacy download en_core_web_sm
# 3. set up path in line 25
# 4. set up path to stop words in line 


import help_NER_UB as help_ner
import numpy as np
import spacy
import scispacy
import en_core_web_sm
# from spacy.lang.en.stop_words import STOP_WORDS
import nltk
import pandas as pd
import time

# use spacy small model for POS
nlp = en_core_web_sm.load()

# use spacy large model for NER _ set up path
# nlp_ner = spacy.load('path_to/scispacy/en_core_sci_lg-0.2.4/en_core_sci_lg/en_core_sci_lg-0.2.4')
# nlp_ner = spacy.load('./en_core_sci_lg-0.2.4/en_core_sci_lg/en_core_sci_lg-0.2.4')
nlp_ner = spacy.load('/Users/mac/anaconda3/lib/python3.7/site-packages/scispacy/en_core_sci_lg-0.2.4/en_core_sci_lg/en_core_sci_lg-0.2.4')

# Set up stopwords
# Need to Change the path
# path_to_stopwords = 'path_to_stopwords/nltk_stopwords.txt'
# path_to_stopwords = '/Users/mac/Documents/GitHub/COVIDGraphProject/nltk_brown_stopwords.txt'
   
# with open(path_to_stopwords, 'r') as file:
#     stopwords = file.readlines()
# for word in stopwords:
#     nlp.vocab[word].is_stop = True

STOP_WORDS = help_ner.getStopwords()

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
            more_subs += [tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"]
            if len(more_subs) > 0:
                more_subs += _get_subs_from_conjunctions(more_subs)
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs +=[tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"]
            if len(more_objs) > 0:
                more_objs+=_get_objs_from_conjunctions(more_objs)
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
            subs+=_get_subs_from_conjunctions(subs)
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
            objs+=[tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')]
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
                    objs += _get_objs_from_prepositions(rights, is_pas)
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
            objs+=_get_objs_from_prepositions(rights, is_pas)
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET" and tok.lower_ not in STOP_WORDS]
    if len(subs) > 0:
        subs+=_get_subs_from_conjunctions(subs)
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs+=foundSubs
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
    objs+=_get_objs_from_prepositions(rights, is_pas)

    #potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs+=potential_new_objs
        v = potential_new_verb
    if len(objs) > 0:
        objs+=_get_objs_from_conjunctions(objs)
    return v, objs


# return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return toks


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
def expand(item, tokens, visited, list_ents_lemma_):
    if item.lower_ == 'that':
        item_res = _get_that_resolution(tokens)
        if type(item_res) != spacy.tokens.doc.Doc:
            item = item_res


    parts = []

    # if hasattr(item, 'lefts'):
    #     for part in item.lefts:
    #         if part.pos_ in BREAKER_POS:
    #             break
    #         if not part.lower_ in NEGATIONS:
    #             parts.append(part)

    node = get_node(item, tokens, list_ents_lemma_)
    if node:
        parts.append(node)

    # if hasattr(item, 'rights'):
    #     for part in item.rights:
    #         if part.pos_ in BREAKER_POS:
    #             break
    #         if not part.lower_ in NEGATIONS:
    #             parts.append(part)

    # if hasattr(parts[-1], 'rights'):
    #     for item2 in parts[-1].rights:
    #         if item2.pos_ == "DET" or item2.pos_ == "NOUN":
    #             if item2.i not in visited:
    #                 visited.add(item2.i)
    #                 parts.extend(expand(item2, tokens, visited))
    #         break
    
    return parts
    

# convert a list of tokens to a string
def to_str(tokens):
    return ' '.join([item for item in tokens])

# converts scibert's dict of result into list of lists
def transpose_list(dict_res):
    list_items = list()
    for key,value in dict_res.items():
        list_items.append(value)
    return np.array(list_items).T.tolist()

def is_in_list_ents_lemma_(token, list_ents_lemma_):
    for ent in list_ents_lemma_:
        if token.lower_ in ent:
            return True


# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens, list_ents_lemma_):
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
                    sub_bool = is_in_list_ents_lemma_(sub, list_ents_lemma_)
                    for obj in objs:
                        # objNegated = _is_negated(obj)
                        if is_in_list_ents_lemma_(obj, list_ents_lemma_) or sub_bool:
                            if is_pas:  # reverse object / subject for passive
                                o = to_str(expand(obj, tokens, visited, list_ents_lemma_))
                                s = to_str(expand(sub, tokens, visited, list_ents_lemma_))
                                if len(o)*len(s)!=0:
                                    svos.append((o,s))
                            # svos.append((to_str(expand(obj, tokens, visited)),
                            #                to_str(expand(sub, tokens, visited)), v2.lemma_ + ' not' if verbNegated or objNegated else v2.lemma_))
                            else:
                                s = to_str(expand(sub, tokens, visited, list_ents_lemma_))
                                o = to_str(expand(obj, tokens, visited, list_ents_lemma_))
                                if len(o)*len(s)!=0:
                                    svos.append((s,o))
                            # svos.append((to_str(expand(sub, tokens, visited)),
                            #               to_str(expand(obj, tokens, visited)), v2.lemma_ + ' not' if verbNegated or objNegated else v2.lemma_))
            else:
                v, objs = _get_all_objs(v, is_pas)
                for sub in subs:
                    sub_bool = is_in_list_ents_lemma_(sub, list_ents_lemma_)
                    for obj in objs:
                        if is_in_list_ents_lemma_(obj, list_ents_lemma_) or sub_bool:
                        # objNegated = _is_negated(obj)
                            if is_pas:  # reverse object / subject for passive
                                o = to_str(expand(obj, tokens, visited, list_ents_lemma_))
                                s = to_str(expand(sub, tokens, visited, list_ents_lemma_))
                                if len(o)*len(s)!=0:
                                    svos.append((o,s))
                            else:
                                s = to_str(expand(sub, tokens, visited, list_ents_lemma_))
                                o = to_str(expand(obj, tokens, visited, list_ents_lemma_))
                                if len(o)*len(s)!=0:
                                    svos.append((s,o))


    return svos

def get_text(file, sep):
    cont = list()
    with open(file, 'r') as file:
        line = file.readlines()
    for abstr_ in line:
        sentences = abstr_.strip().split(sep)
        cont+=[sen_ for sen_ in sentences[1:]  if (sen_ != '')&(~sen_.isnumeric())]
    return cont

# def extract_nodes(sentence):
#     list_nodes = list()

#     for word in sentence:
#         if word.lower_ in STOP_WORDS:
#             pass
#         rights = list(word.rights)
#         lefts = list(word.lefts)
#         if word.pos_ == 'NOUN':
#             for tok in lefts:
#                 if tok.dep_ == 'compound' or tok.dep_ == 'amod' or tok.pos_ == 'NUM': # to be refined
#                     if tok.lower_ not in STOP_WORDS:
#                         list_nodes.append(tok.lemma_+' '+word.lemma_)
#                     else:
#                         list_nodes.append(word.lemma_)
#                 else:
#                     list_nodes.append(word.lemma_)
#     return list_nodes



def get_node(token, tokens, list_ents_lemma_):
    if (help_ner.checkEntityForNumeric(token.lower_)) or (token.lower_ in STOP_WORDS):
        return None
 
    
    for ent in list_ents_lemma_:
        if token.lower_ in ent:
            return ent
    
    # lefts = list(token.lefts)            
    # if len(lefts)>0:
    #     for left in lefts:
    #         if left.dep_ == 'compound' or left.dep_ == 'amod' or ((token.lower_ == '%') and (left.pos_ == 'NUM')): # to be refined
    #             return (left.lemma_+' '+token.lemma_)
    #         else:
    #             return token.lemma_
    else:
        return token.lemma_

 


def extract_link(sen):
    list_edges = list()
    list_ents = nlp_ner(sen.text).ents
    list_ents_lemma_ = [e.lemma_ for e in list_ents]
    # print(list_ents_lemma_)
    
    res = list()

    if len(list_ents_lemma_):
    
        preps = [tok for tok in sen if tok.pos_ == "ADP" and tok.dep_ == "prep"]    # covers the cases where a verb has no subject?
        for prep in preps:
#         lefts = [tok for tok in prep.lefts if tok.pos_ == 'NOUN']
            left = prep.head
            rights = [tok for tok in prep.rights if tok.pos_ == 'NOUN']
            if len(rights)>0:
                if is_in_list_ents_lemma_(left, list_ents_lemma_) or is_in_list_ents_lemma_(rights[0], list_ents_lemma_):
                    left_node = get_node(left,sen, list_ents_lemma_)
                    right_node = get_node(rights[0],sen, list_ents_lemma_)
                    if (left_node is not None) and (right_node is not None):
                        list_edges+= [(left_node,right_node)]

    

        list_edges += findSVOs(sen, list_ents_lemma_)


        res = [(s,o) for (s,o) in list_edges if (s !='') and (o !='') and (s != o)]
    # print(res)
    return res 

# def get_sentences(curr_abstract):
#     sentences = nltk.tokenize.sent_tokenize(curr_abstract)
#     #cont = [sen_ for sen_ in sentences[1:]  if (sen_ != '')&(~sen_.isnumeric())]
#     return sentences

def create_df_sen(data, name_col = 'full_text'):
    list_rec = list()
    for ix, row in data.iterrows():
        list_sen = [x for x in row[name_col].split(' . ') if x!= '']
        for sen in list_sen:
            list_rec.append([row['sha'], row['fullname'], sen, row['year'], row['date']])
    
    df = pd.DataFrame.from_records(list_records, columns = ['sha', 'fullname', 'sentence', 'year', 'date']) 
    df['links'] = ''
    
    return df



def main():
    all_text = "however subsequent studies pioneered by arm and coworkers have shown that in several situations pla2g5 exerts antiinflammatory functions which may rely on a common mechanism involving the regulation of macrophage phagocytosis by this spla2"
    # all_text = 'although differences in animal housing and pathogen exposure could account for the phenotypic differences found in these studies it has been suggested that the strategies used to make the knockout mice in these studies differed and some of the th17-axis inflammatory phenotype could be due to the synthesis of a truncated trim21 form lacking its c-terminal half'
    links = extract_link(nlp(all_text))

    print(links)
    

if __name__ == '__main__':
    main()



