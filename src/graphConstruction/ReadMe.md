# Ulya Bayram
# ulyabayram@gmail.com
#
import langdetect

# Takes full text as input, returns True if text is English, False otherwise
def isEnglish(text_):

    try:
        lang_ = langdetect.detect(text_)
    except langdetect.lang_detect_exception.LangDetectException:
        return False
    else:
        if  lang_ == 'en':
            return True
        else:
            return False

# Returns True if the input entity is numeric - in which case we should ignore this entity
# False - is what we want
def checkEntityForNumeric(curr_entity):
    curr_entity = curr_entity.replace(' ', '')
    curr_entity = curr_entity.replace('%', '')
    curr_entity = curr_entity.replace(',', '')
    curr_entity = curr_entity.replace('.', '')

    return curr_entity.isnumeric()

# Just call this function to get the list of unique stopwords
def getStopwords():
    fo = open('nltk_brown_stopwords.txt', 'r')
    list_words = fo.read().split('\n')[:-1]

    return list(set(list_words))