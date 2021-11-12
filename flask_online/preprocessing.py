import re
import jieba
import zhconv

def tr_to_sim(text):
    res = zhconv.convert(text, "zh-cn")
    return res

def jieba_cut_text(text, stop_word, cut_by=" "):

    text = tr_to_sim(text)
    # [\u4E00-\u9FA5]+ 去除非中文字符
    patt = re.compile("[\u4E00-\u9FA5]+", re.VERBOSE|re.IGNORECASE)
    word_list = []
    for t in patt.findall(text):
        word_list += [word for word in jieba.cut(t) if word not in stop_word]
    return cut_by.join(word_list)