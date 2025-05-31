import difflib
word1 = "byebye"
word2 = "Bye-bye!"
print(difflib.SequenceMatcher(None, word1.lower(), word2.lower()).ratio())