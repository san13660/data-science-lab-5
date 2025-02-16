import re
import sys
import pandas as pd
import numpy as np
import collections

def process_line(line):
    processed = line.lower()

    processed = re.sub(r'''((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))''', '', processed)

    processed = processed.replace('#','')
    processed = processed.replace('@','')
    processed = processed.replace("’re",' are')
    processed = processed.replace("n’t",' not')
    processed = processed.replace("’ve",' have')
    processed = processed.replace("’ll",'')
    processed = processed.replace("’s",'')
    processed = processed.replace("’d",'')
    processed = processed.replace("’m",' am')
    
    processed = processed.replace('_',' ')
    processed = processed.replace('-',' ')
    processed = processed.replace('–',' ')
    processed = processed.replace('—',' ')
    processed = processed.replace("/",' ')

    whitelist = set('abcdefghijklmnopqrstuvwxyz ')
    processed = ''.join(filter(whitelist.__contains__, processed))

    processed = ' ' + processed + ' '

    processed = processed.replace(' a ',' ')
    processed = processed.replace(' of ',' ')
    processed = processed.replace(' to ',' ')
    processed = processed.replace(' the ',' ')
    processed = processed.replace(' and ',' ')
    processed = processed.replace(' for ',' ')
    processed = processed.replace(' but ',' ')
    processed = processed.replace(' or ',' ')
    processed = processed.replace(' yet ',' ')
    processed = processed.replace(' so ',' ')
    processed = processed.replace(' as ',' ')
    processed = processed.replace(' either ',' ')
    processed = processed.replace(' nor ',' ')
    processed = processed.replace(' until ',' ')
    processed = processed.replace(' that ',' ')
    processed = processed.replace(' which ',' ')
    processed = processed.replace(' where ',' ')
    processed = processed.replace(' while ',' ')
    processed = processed.replace(' i ',' ')
    processed = processed.replace(' in ',' ')
    processed = processed.replace(' it ',' ')
    processed = processed.replace(' is ',' ')
    processed = processed.replace(' you ',' ')
    processed = processed.replace(' with ',' ')
    processed = processed.replace(' was ',' ')
    processed = processed.replace(' on ',' ')
    processed = processed.replace(' my ',' ')
    processed = processed.replace(' not ',' ')
    processed = processed.replace(' this ',' ')
    processed = processed.replace(' have ',' ')
    processed = processed.replace(' are ',' ')
    processed = processed.replace(' be ',' ')
    processed = processed.replace(' we ',' ')
    processed = processed.replace(' at ',' ')
    processed = processed.replace(' he ',' ')
    processed = processed.replace(' from ',' ')
    processed = processed.replace(' all ',' ')
    processed = processed.replace(' they ',' ')
    processed = processed.replace(' me ',' ')
    processed = processed.replace(' one ',' ')
    processed = processed.replace(' by ',' ')
    processed = processed.replace(' do ',' ')
    processed = processed.replace(' about ',' ')
    processed = processed.replace(' will ',' ')
    processed = processed.replace(' what ',' ')
    processed = processed.replace(' up ',' ')
    processed = processed.replace(' out ',' ')
    processed = processed.replace(' his ',' ')
    processed = processed.replace(' an ',' ')
    processed = processed.replace(' if ',' ')
    processed = processed.replace(' had ',' ')
    processed = processed.replace(' her ',' ')
    processed = processed.replace(' when ',' ')
    processed = processed.replace(' there ',' ')
    processed = processed.replace(' just ',' ')
    processed = processed.replace(' like ',' ')
    processed = processed.replace(' your ',' ')
    processed = processed.replace(' can ',' ')
    processed = processed.replace(' she ',' ')
    processed = processed.replace(' has ',' ')
    processed = processed.replace(' more ',' ')
    processed = processed.replace(' their ',' ')
    processed = processed.replace(' some ',' ')
    processed = processed.replace(' who ',' ')
    processed = processed.replace(' would ',' ')
    processed = processed.replace(' our ',' ')
    processed = processed.replace(' were ',' ')
    processed = processed.replace(' them ',' ')
    processed = processed.replace(' been ',' ')
    processed = processed.replace(' the ',' ')
    processed = processed.replace(' its ',' ')
    processed = processed.replace(' get ',' ')
    processed = processed.replace(' no ',' ')
    processed = processed.replace(' am ',' ')
    processed = processed.replace(' how ',' ')
    processed = processed.replace(' also ',' ')
    processed = processed.replace(' us ',' ')
    processed = processed.replace(' did ',' ')
    processed = processed.replace(' than ',' ')
    processed = processed.replace(' him ',' ')
    processed = processed.replace(' could ',' ')
    processed = processed.replace(' very ',' ')
    processed = processed.replace(' much ',' ')
    processed = processed.replace(' these ',' ')
    processed = processed.replace(' into ',' ')
    processed = processed.replace(' then ',' ')
    processed = processed.replace(' because ',' ')

    processed = " ".join(processed.split())
    return processed

df = pd.read_csv('GrammarandProductReviews.csv', dtype=object)

df['reviews.text'].replace('', np.nan, inplace=True)
df.dropna(subset=['reviews.text'], inplace=True)

wordcount = {}

for index, row in df.iterrows():
    df.at[index,'reviews.text'] = process_line(row['reviews.text'])

    for word in row['reviews.text'].split():
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

df.to_csv("preprocesado_GrammarandProductReviews.csv")

print('Archivo preprocesado guardado (preprocesado_GrammarandProductReviews.csv)')

n_print = 10
word_counter = collections.Counter(wordcount)
lst = word_counter.most_common(n_print)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.to_csv("frecuencia_palabras.csv")

print('Archivo de frecuencia de palabras guardado (frecuencia_palabras.csv)')