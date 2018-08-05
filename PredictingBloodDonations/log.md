Day 1 : 7/6/18

Today's progress: Looked for and found a competition to participate in, a warm up competition by DrivenData called 'Predict Blood Donations'. Downloaded data into csv files. Started code with import statements.


Thoughts: I chose this competition as I am not very experienced with ML, and want to learn more. It is simple, but I hope to do well.      

Day 2: 7/7/18

Today's progress: I got a lot done today. I made the train, dev, and test sets. I made a KNN classifier, and trained it on the basic data
to receive a plain accuracy score of about 85 percent. I tried to use a scatterplot to find correlations between the data and y, but I
could not use the scatterplot intuitively for binary graphing. I will find better analysis methods tomorrow.

Thoughts: I got better at using classification, and I really am excited to figure out how to use the P-score and CAP curves tomorrow
to aid in my progress.


Day 3: 7/8/18

Today's progress: I learned how to use the CAP curve and P-scores, but could not derive proper results in my code. I learned how to use PCA, and attempted to apply it to my code. Then, I found the (current) optimal number of neighbors for the KNN algorithm using a loop.

Thoughts: PCA is quite interesting. I hope I can find more tools to aid my model, but I am running out of ideas which are not hyperparameter tuning.


Day 4 : 7/11/18

Today's progress: I took a little break to work on another DialogFlow project, but I am returning to this project today. I tried to find ways to optimize my KNN algorithm, but was unsuccessful. I decided to try out different models : KNN, Naive Bayes, SVM with a linear kernel, SVM with a gaussian kernel, and Random Tree Classification. In the end, SVM with a linear kernel, using only 3 features(cut down using PCA) returned the highest accuracy of 90.475%.

Thoughts: Now that I have chosen a new model, I would love to see if I can optimize this one further.


Day 5: 7/13/18

Today's Progress: I tested out a neural network approach to the problem. It yielded seemingly random results, ranging from ~80% to 92% at one time. I attempted hyperparameter tuning, but decided my dataset was a little too small to be using a neural network on. I only have 450 training examples and 3 features, and I believe this is too little data for a neural network, no matter the architecture, and resorted to my SVC. I attempted to tune some hyperparameters, such as the shrinking rate, but none really applied, so I just applied the shrinking rate and random state to help with further testing.

Thoughts: I cannot see any more optimization methods I can apply at this point. I might consider trying out different models next time.

Day 6: 7/14/18
Today's Progress: After a lot of searching, I was unable to find good improvement methods. When attempting to submit my file, I noticed that the SVC classifier was outputing a 0, or false, for every piece of test data. I now am going back and attempting to find the reason for this, as well as tuning the hyperparameters. 

Thoughts: None


Day 7 : 7/16/18
Today's Progress : I tried a new kernel for the SVC model, poly, and it returned the same accuracy as my linear model. I tuned the hyperparameters, and found out that the C-score does not change the linear or poly model. The degree does not affect accuracy when it is in the range (1-4), but once it surpasses this range, overfits the training set and has lower accuracy than a model with a smaller degree. The gamma hyperparameter also did not affect the model for the better or worse, but results in huge run-times for the model as it increases. The shrinking hyperparameter does not affect the model. Neither does the tolerance level, except when set to values higher than 2 - decreases the model's performance. Finally, the coefficient value. When negative, it negatively impacts the model performance, but when positive, and in a reasonable value(i.e 1-40), slightly helps the model performance.

To summarize: In my final model, I used the SVM classifier with a polynomial kernel, a degree of 3(default), and a coefficient of 1. Once I submitted the model to Driven Data, I recieved a 90.8 % accuracy, which is, in my opinion, a decent score for a beginner like me.

Thoughts: After I take some time off to work on my website, I would love to jump back in and start a new project in DrivenData or another ML site. This was an amazing process that taught me a lot about cross-validation, confusion matrices, classification models, hyperparameters, and much more. I have truly enjoyed it, even though my model might have been buggy at times.
