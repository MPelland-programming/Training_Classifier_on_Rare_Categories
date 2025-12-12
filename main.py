import os
import pandas as pd

#Get API key
apikey = os.environ["ANTHROPIC_API_KEY"]

#for pandas dataset rawdata, provide first 10 rows for which the value of the first column is 1
rawdata = pd.read_csv("data/dataset.csv")
data_subset = rawdata[rawdata.iloc[:, 0] == 1].head(10).iloc[:,3]

#concatenate all elements of data_subset into a single string with each element separated by Question:
q_context = " Question: ".join(data_subset.astype(str).tolist())


#using transformers to tokenize dataset and then obtain the non padded lenght of each tokenized phrase

#Loop over all topics to get context.
for tt in list(topic_dict.keys()):
  # Get questions
  data_subset = rawdata[rawdata.iloc[:, 0] == tt].head(15).iloc[:,3]

  # Generate context for batch processing.
  full_context = "Question: "+" Question: ".join(data_subset.astype(str).tolist())
  topic = topic_dict[tt]
  print(topic)
  print(len(full_context))
  print(full_context)

  # dictionary of topics.
  topic_dict = {
      1: "Society and Culture"
      , 2: "Science and Mathematics"
      , 3: "Health"
      , 4: "Education and Reference"
      , 5: "Computers and Internet"
      , 6: "Sports"
      , 7: "Business and Finance"
      , 8: "Entertainment and Music"
      , 9: "Family and Relationships"
      , 10: "Politics and Government"
  }

#tokenize all train data to get a distribution of token lenghts.
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

len_ds = Dataset.from_pandas(rawdata.iloc[:,3].to_frame(name="text").astype(str))
opts = {'padding': False, 'truncation': True, 'max_length': 512}

def tokenize_function(examples):
    return tokenizer(examples["text"], **opts)

tokenized_datasets = len_ds.map(tokenize_function, batched=True)


