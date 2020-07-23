import math
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer


testing = """

Microsoft will stop relying on Amazon to help it run the popular Minecraft video game.

The shift represents an obvious way for Microsoft to cut back on payments to one of its toughest competitors and promote its own product. Amazon Web Services rules the market for public cloud infrastructure for running software from afar through vast data centers, and Microsoft has been working to take share with its Azure cloud.

Azure is growing faster than many other parts of Microsoft, helping it lean less on longstanding properties like Windows and Office. Moving more of its own software to Azure can help Microsoft make the case to customers that it doesn’t look anywhere else for computing, storage and networking resources to deliver its online services. That’s an important consideration, because Amazon can tell customers that its sprawling e-commerce business consumes resources from AWS.

Most of Microsoft’s consumer and commercial properties, including the Teams communication app, already draw on Azure. Last year, two and a half years after it closed the acquisition of business social network LinkedIn, Microsoft said it would migrate LinkedIn from its own dedicated data centers to Azure. 

The use of AWS for Minecraft for a version called Realms — virtual places for small groups to gather and play the open-world game together — dates to 2014. Months after AWS published a blog post about how Mojang, the game developer behind Minecraft, had chosen to tap AWS for Realms, Microsoft announced that it would acquire Mojang for $2.5 billion.

Minecraft has since grown into the world’s best-selling game, with over 200 million copies sold as of May, and 126 million people play it each month.

“Mojang Studios has used AWS in the past, but we’ve been migrating all cloud services to Azure over the last few years,” a Microsoft spokesperson told CNBC in an email. Amazon declined to comment.

It would not have been right to make Mojang get off AWS immediately after the acquisition, Matt Booty, the head of studios at Microsoft, suggested in a recent interview.

“It would be easy for a large organization to come in and say: ‘Hey, we’re going to show you how it’s done. We’re going to get you off this Java code. We’re going to get things moved over to C. We’re going to get you off Amazon Web Services and over to Azure,’” Booty told GamesIndustry.biz. “But it’s important to realize that the conditions that created Minecraft, how it came to be, are likely to be things that are difficult to recreate within a more corporate structure.”

Now there is an end in sight for the dependence on a rival.

“We’ll be fully transitioned to Azure by the end of the year,” the Microsoft spokesperson wrote.

"""


#A combination of preprocessing and tokenization
#RETURN: A double array of stemmed tokens
def pre_token(s):
    sentence_list = sent_tokenize(s)
    words_all = {}
    all_sentences = []
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    for sentence in sentence_list:
        word_list = (word_tokenize(sentence))
        one_sentence = []
        for x in word_list:
            x = x.lower()
            if x in stop_words:
                continue
            x = stemmer.stem(x)
            if (len(x) == 1):
                continue
            one_sentence.append(x)
        all_sentences.append(one_sentence)
    #print(all_sentences)
    return all_sentences


def tf(doub_array):
    holding = doub_array
    working_freq = []
    word_counts = []
    for grouping in doub_array:
        total_terms = 0
        for word in grouping:
            total_terms += 1  # We now have the total number of terms in each sentence
        word_counts.append(total_terms)
        #print(total_terms)
    zipped = zip(word_counts, holding)

    for line in zipped:
        word_freq = {}
        for word in line[1]:
            if word in word_freq:
                word_freq[word] += (1/line[0])
            else:
                word_freq[word] = (1/line[0])
        working_freq.append(word_freq)
    #print(working_freq)
    return working_freq


def idf(double_array):
    idf_dic = {}
    idf_log = {}
    sentence_count = 0
    for sentence in double_array: #Number of sentences
        sentence_count += 1
    for sentence in double_array: #Number of sentences containing word
        for word in sentence:
            if word in idf_dic:
                idf_dic[word] += 1
            else:
                idf_dic[word] = 1
    for x in idf_dic:
        #print(idf_dic.get(x))
        idf_log[x] = math.log(sentence_count/idf_dic.get(x))
    #print(idf_log)
    return idf_log

def multiply(tf, idf):

    new_document = []
    for sentence in tf:
        new_sentence = []
        for word in sentence:
            new_sentence.append([word,sentence.get(word)])
        new_document.append(new_sentence)
    #print(new_document) #in good working order


    for sentence in new_document:
        for word in sentence:
            word[1] = idf.get(word[0]) * word[1]
        #print(sentence)
    #print(new_document)
    return new_document

def cumulate(trip_array):
    all_sentence_score = []
    for sentence in trip_array:
        sent_total = 0
        new_sentence = []
        for word in sentence:
            sent_total += word[1]
            new_sentence.append(word[0] + " ")
        all_sentence_score.append([sent_total, new_sentence])
    print(all_sentence_score)
    return all_sentence_score




a = pre_token(testing)
b = tf(a) #Line by line document, dictionary of term freq, b[0] is first 10 rods, b[1] is next 13, etc
print()
print()
c = idf(a)
d = multiply(b,c)
cumulate(d)
