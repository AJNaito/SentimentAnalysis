import sys
from SentimentalModel import SentimentalModel
import pandas as panda

def ProgressBar(iter, total, length = 50):
    prefix = "Progress:"
    end = ("{0:." + str(2) + "f}").format(100 * (iter/float(total))) # Current progress
    fixedLength = int(length * iter / total)
    bar = fixedLength * "■" + "-" * (length - fixedLength)
    print (f"{prefix} |{bar}| {end}% |Number of Items Completed: {iter}")
    sys.stdout.write("\033[F")

def ComputeAccuracy():
    colnames = ['polarity', 'id', 'post_datetime', 'query', 'user', 'tweet']
    with open("training/training.1600000.processed.noemoticon.csv", "r") as csvfile:
        count = 0
        correct = 0
        dataSet = panda.read_csv(csvfile, names = colnames)
        dataSet = dataSet[["polarity", "tweet"]]
        total = len(dataSet)

        for index,row in dataSet.iterrows():
            prediction = model.PredictSentiment(row["tweet"])
            if row["polarity"] == 4 and prediction == "Positive":
                correct += 1
            
            elif row["polarity"] == 0 and prediction == "Negative":
                correct += 1

            count += 1
            ProgressBar(count, total, 50)
        
        return (100 * (correct/float(total)))

print("Training Model...")
model = SentimentalModel()
print("Done!")

if model.valid:
    print("Computing Model Accuracy...")
    percentage = ComputeAccuracy()
    print("")
    print(f"This model has an accuracy of: {percentage}")

    print("Type a statement, enter q to quit")
    while 1:
        line = input()
        if "q" == line.rstrip():
            quit()

        prediction = model.PredictSentiment(line)
        print("My best guess is that the statement is " + prediction)
        print("")
else:
    print("No Training Data")