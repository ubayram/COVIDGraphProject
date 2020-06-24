# Ulya Bayram
# ulyabayram@gmail.com
#
import langdetect

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

def checkEntityForNumeric(curr_entity):
    curr_entity = curr_entity.replace(' ', '')
    curr_entity = curr_entity.replace('%', '')
    curr_entity = curr_entity.replace(',', '')
    curr_entity = curr_entity.replace('.', '')

    return curr_entity.isnumeric()