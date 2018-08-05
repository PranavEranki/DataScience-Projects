Day 1: 7/19/18

Today, I found a good problem to work on - a twitter sentiment analysis model from AnalyticsVidhya.com
I have gathered the test and train data sets, reshaped them, and done the initial preprocessing. I have used stopwords,
stemmed the words, and rejoined them, constructing a new corpus (the same size as the train data but processed). Then, I split
the training data into a dev and train set. Next time, I want to train the model and begin LSA.

Day 2: 7/20/18

Today, I fixed a lot of bugs with my code. I learnt more about vectors and matrices. Then, I learned about how LSA works, 
and added PCA to my model. I trained a very basic GaussianNB model to achieve around 75% accuracy, which 
I definetly hope to improve. Next time, I will experiment more with the PCA values, try a KNN classifier, and utilize more tools to
improve the NLP model.

Day 3: 7/22/18

I forgot to commit my changes yesterday, so I will commit + describe them today. I organized my code a bit, then tried a couple different PCA values for the n_components, then settled on 350. Then, I used a KNN classifier and trained to model to recieve slightly higher results. I then began to read about how to use Shuffle Split and cross val scores to test my models performance. After a few iterations of testing hyperparameters, I settled on 9 neighbors for the KNN classifier. Around this point, I decided that the hyperparameter choosing was going too slow. I began to research how to find hyperparameters fast and effectively. I came across a couple Medium articles on how to use NLP with sklearn's GridSearchCV. I read up on how to use this, made lists of the different parameters I wanted to use, then trained the GridSearchCV. The first few times, I got errors about taking the square root of a negative number somewhere in my code. After experimenting with the hyperparameters list, I realized that for some reason, with different values of P, I come across this error. I then threw P out of the hyperparameter list, atleast temporarily. Now, I began to come across another error - an Attribute Error. After nearly 2 hours of attempting to debug this error, I gave up and posted my question on Stack Overflow. 

Link to the question : https://stackoverflow.com/questions/51462538/always-getting-attribute-error-when-using-gridsearchcsv-with-knn

I then set the model to run on 1 core around midnight, then left it to run.

Da 4: 7/23/18

I woke up this morning to find the same error message in my console. I updated my Stack Overflow question, adding the training data and the error message. While waiting for a response, I made a new file which organized my code into multiple methods. I then wrote a main method. I am quite happy with this result, as this is the cleanest code I have wrote so far, and it still provides crisp explanations of the methods. As of 10:30 in the morning, I am waiting for the question to be answered. Once it does, I will be sure to update my code, fix the errors, and confirm it works before commiting the changes and submitting my code to the challenge page.


Day 5: 7/26/18

I have been working on this problem on and off for the past few days. The stack overflow question I posed has recieved a lot of help, and over the course of a few days, I have slowly solved all the errors. I have also found out something quite important:

When GridSearch is imported from sklearn.model_selection, you often run into a memory error. However, when it is imported from sklearn.grid_search, I do not run into this error. I believe this occurs mainly on Windows. My python version is 64 bit also, so there was no problem correlating to 32-bit python. I have found this issue on sklearn, and I hope to find a fix.

My gridSearch took too much time, as described in the ReadMe, so instead I decided to go with a course-to-fine approach. I started with a colossal list of parameters, and trained the model on a few random choices. I found the minimum error to be around 12 neighbors and a leaf size of 50. I removed any unnecessary parameters, including P and the algorithm lists.
I started from here, and took multiple different GridSearches and narrowed it down. Here is the generic iteration information.

ITERATION 1
*leaf size - 50
*neighbors - 18
*weight - distance

ITERATION 2
*leaf size - 45
*Neighbors - 20

ITERATION 3
*Leaf size - 35
*Neighbors - 16
*Weighted on f1 score

Now, I have found the best parameters I could. I update my Final Code to have a 'use default' option, which just fits the model with the default best parameters I found for my problem.

Once I finish the model and all the code is finalized, I run it, and get a predictions.csv.
Oddly, no matter how I change it, I am unable to submit this file.
I reckon I would have likely gotten about 88 - 90 percent based on accuracy on dev set.
I have included the predictions file in the code and data section.
In case you can find a way to fix it and upload, please be my guest and do so. If you can tell me your score, I will be delighted to know it. 

Thanks, and see you later,

Pranav
