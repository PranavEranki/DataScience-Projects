## Twitter Hate Speech Detector

Problem from Analytics Vidhya which takes tweets and detects if they are, or employ, hate speech. 

Problem link - https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/

<hr/>

### Reason
I wanted to get more machine learning experience, so I tried this problem from AnalyticsVidhya. 
It is more complex than previous problems I have attempted, so I am excited to see how I do.
I have also never attempted many NLP problems, so this would be a nice change from the standard classification problems I attempt.

<hr/>

### Code + Modules
The code is written in Python, using primarily NLTK, pandas, and sklearn. The classifier is a KNN classifier. I wrote the code using the Spyder Code Editor from conda, as it is convenient, simple, and provides a nice file organization, in my opinion.

<hr/>

### Contents
The code is organized into functions for reusability. Just input the correct parameters, modify the classifier / parameter lists for the GridSearchCv if necessary, and the code should be reusable.



### GRID SEARCH -- IMPORTANT

The initial gridSearch I coded had a run time of approximately 10 hours - too long, in my opinion. It also carried with it a seemingly endless array of issues, including Memory Errors, Attribute Errors, you name it. After a long and hard discussion on stack Overflow with some extremely helpful members, I managed to solve the problem - only to be faced with the issue of an insanely long run-time. Instead, I decided to go with a coarse to fine model approach. Train the parameters on a broad set of data, then slowly refine them until I found the optimum parameters for an F1 Scoring system. You can view a more detailed process explanation in the Log.md file, but if you are to reuse this code, please go through your own coarse-to-fine approach with your own parameter lists and customizations of choice - no need to follow mine.

Thank you.
