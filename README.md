# SENTIMENTAL-ANALYSIS-AMAZON-PRODUCT-BASED-REVIEWS
-----

# **Project Overview:**
This project analyzes sentiment in Amazon Fine Food Reviews using various techniques, including VADER (Valence Aware Dictionary and sEntiment Reasoner) and the RoBERTa model from Hugging Face's Transformers library. The goal is to evaluate the sentiment of customer reviews and visualize the results to gain insights into customer opinions.

# **Table of Contents:**
```bash
1. Dataset
2. Technologies Used
3. Installation
4. Usage
5. Results
6. Contributing
7. License
```
----

# **Dataset:**

The dataset used for this analysis is the Amazon Fine Food Reviews dataset, which contains reviews of fine foods from Amazon. You can download it from Kaggle.


# **Technologies Used:**
```
1. Python: The programming language used for the analysis.
2. Pandas: For data manipulation and analysis.
3. NumPy: For numerical operations.
4. Matplotlib: For creating visualizations.
5. Seaborn: For statistical data visualization.
6. NLTK: For natural language processing tasks.
7. Transformers: For using pre-trained models like RoBERTa.
8. TQDM: For progress bars in loops.
```

# **Installation:**
```bash
To set up the project, make sure you have Python installed. You can install the required libraries using pip:
pip install pandas numpy matplotlib seaborn nltk transformers tqdm
```

# **Usage:**

1. Import Libraries: Start by importing the necessary libraries in your Python script or Jupyter Notebook or Kaggle Notebook.
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.notebook import tqdm
```

2. Load Data:
 ```bash
   df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
   df = df.head(500)  # Limit to the first 500 reviews for analysis
```

3. Data Exploration:
```bash
   print(df.shape)
   ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(11, 6))
   ax.set_xlabel('Review Stars')
   plt.show()
```

4. Sentiment Analysis with VADER: Use NLTK's VADER to analyze sentiment.
```bash
   from nltk.sentiment import SentimentIntensityAnalyzer
   sia = SentimentIntensityAnalyzer()
   example = df['Text'][50]
   print(sia.polarity_scores(example))
```

5. Sentiment Analysis with RoBERTa: Use the RoBERTa model for sentiment analysis from Hugging Face.
```bash
   MODEL = "/kaggle/input/twitter-roberta-sentiment-ananlysis/transformers/default/1"
   tokenizer = AutoTokenizer.from_pretrained(MODEL)
   model = AutoModelForSequenceClassification.from_pretrained(MODEL)

   def polarity_scores_roberta(example):
       encoded_text = tokenizer(example, return_tensors='pt')
       output = model(**encoded_text)
       scores = output[0][0].detach().numpy()
       return {
          'roberta_neg': scores[0],
          'roberta_neu': scores[1],
          'roberta_pos': scores[2]
        }
```

6. Combine Results: Combine results from both VADER and RoBERTa for a comprehensive analysis.
```bash
     res = {}
     for i, row in tqdm(df.iterrows(), total=len(df)):
         text = row['Text']
         myid = row['Id']
         vader_result = sia.polarity_scores(text)
         roberta_result = polarity_scores_roberta(text)
         res[myid] = {**vader_result, **roberta_result}
```

7. Visualization: Visualize the sentiment scores.
```bash
results_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

sns.pairplot(data=results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'])
plt.show()
```

# **Example Output:**
```bash
  [
  {
    "label": "POSITIVE",
    "score": 0.9998
  }
 ]
```

# **Running on Kaggle:**
```bash
1. Upload the pretrained model as a dataset in Kaggle.
2. Find the correct MODEL_PATH using !ls commands.
3. Run the Python script to analyze sentiment.
```

# **License:**
```bash
Available under the MIT License.
```





