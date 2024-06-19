import re

def ReduceWordLength(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def PreprocessText(tweet:str):
    #lower case all letters (avoids special cases nIce vs nice)
    temp = tweet.lower()

    #remove unnecessary text (mentions, hashtags, links, and numbers)
    # We are only interested in the words
    temp = re.sub(r"@[A-Za-z0-9]+", "", temp)
    temp = re.sub(r"#[A-Za-z0-9]+", "", temp)
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub(r"[0-9]", "", temp)
    temp = re.sub(r",*(\s*\b(?:and|the|can|you|our|because|such|who))\b", "", temp)

    # Remove quotations
    temp = re.sub(r"['\"]", "", temp)

    #split the tweet into individual words
    temp = temp.split(" ")

    # remove any unnecessary letters in a word (love vs loooooveee)
    temp = [ReduceWordLength(word) for word in temp]

    # remove short words 
    # short words like 'a' will be frequent and don't have any sentiment attached to it
    temp = [word for word in temp if (len(word) > 2)]
    
    # rejoin tweet for training
    return " ".join(word for word in temp)
    