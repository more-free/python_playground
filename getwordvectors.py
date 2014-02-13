__author__ = 'morefree'

import feedparser
import re

def getwordcounts(url):
    blog_info = feedparser.parse(url)
    wc = {}

    for e in blog_info.entries:
        summary = e.summary if 'summary' in e else e.description

        words = getwords(e.title + ' ' + summary)
        for word in words:
            wc.setdefault(word, 0)
            wc[word] += 1

    return blog_info['feed']['title'], wc


def getwords(html):
    txt = re.compile(r'<[^>]*>').sub('', html)

    words = re.compile(r'[^A-Z^a-z]+').split(txt)

    ##  before return, here we can also remove stop words, remove low frequency words, etc.
    return [word.lower() for word in words if word != '']

apcount = {}
wordcounts = {}
feedlist = [line for line in file('feedlist')]

for feedurl in feedlist:
    title, wc = getwordcounts(feedurl)
    wordcounts[title] = wc
    for word, count in wc.items():
        apcount.setdefault(word, 0)
        apcount[word] += 1


wordlist = [w for w, bc in apcount.items() if 0.5 > float(bc)/len(feedlist) > 0.1]
