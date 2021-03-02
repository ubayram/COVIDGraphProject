#Author: Ulya Bayram
#email : ulya.bayram@comu.edu.tr
#
#------------------------------------------------------------------------------------------------------
#
#The content of this project is licensed under the MIT license. 2021 All rights reserved.
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
#and associated documentation files (the "Software"), to deal with the Software without restriction, 
#including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
#and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
#subject to the following conditions:
#
#Redistributions of source code must retain the above License notice, this list of conditions and 
#the following disclaimers.
#
#Redistributions in binary form must reproduce the above License notice, this list of conditions and 
#the following disclaimers in the documentation and/or other materials provided with the distribution. 
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
#LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
#IN NO EVENT SHALL THE CONTRIBUTORS OR LICENSE HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
#WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
#OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
#
#------------------------------------------------------------------------------------------------------
#
#These code are writen for a research project, published in OIR. If you use any of them, please cite:

#Ulya Bayram, Runia Roy, Aqil Assalil, Lamia Ben Hiba, 
#"The Unknown Knowns: A Graph-Based Approach for Temporal COVID-19 Literature Mining", 
#Online Information Review (OIR), COVID-19 Special Issue, 2021.
#
#------------------------------------------------------------------------------------------------------
# Simple functions to help support the node edge extraction processes
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
