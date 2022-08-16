# Project 3 - NLP Sentiment Analysis: Russia & Ukraine War, Tweets Analysis
*By Freda Qifei Yuan*

## Description
### Objective
On February 24th, 2022, Russian President Valdimir Putin initiated the invasion of Ukraine, calling the attack a "special military operation". Many people used social media to join the conversation. This project analyzes the public's changing response to the conflict in Ukraine over time. Twitter is used as a tool to gauge large scale opinions and sentiment. 

### Dataset
- The dataset used a Twitter streaming API to collect posts in real time. Data features include hashtags, daily tweet frequency, user's geography, and basic dataset statistics. From February 24th to March 27th, there are 200K average daily volume of tweets from a total of 900K twitter users. 

- For this project, three days are chosen to represent three time stages (3rd day, 10th day, and 30th day after the attack started)

- Citation: Haq, Ehsan-Ul, et al. "Twitter Dataset for 2022 Russo-Ukrainian Crisis." arXiv preprint arXiv:2203.02955 (2022)

## Methodology
- Natural Language Processing: performed Sentiment Analysis using Python nltk package
  - SentimentIntensityAnalyzer calculates positivity score
  - Textblob calculates subjectivity score 
  
- Text Preprocessing
  - Filtered to English tweets only
  - Cleaned tweets to remove non-character contents (emoji)
  - Removed stop words
  - Stemmed tweets
  
- General discriptive analysis
  - The most popular hashtags each day
  - Tweets length for positive vs. negative and subjective vs. objective
  - Geography differences: USA, Ukraine, vs. other countries response
  
- Data Visualization
  - Word Cloud 
  - Distribution density plot
 
## Conclusion
- Tweet sentiment moved from subjective but hopeful (3rd day), to negative and objective (10th day), and finally settled in middle ground (30th day).
- Tweet content shifted from "so much gradtitude and intial shock", to "call for humanitarian action", and finally "taking a side: #standwithukraine."
- Tweet Length:
  - Negative Tweets tend to be shorter than positive ones
  - Subjective Tweets tend to be longer compared to objective ones
