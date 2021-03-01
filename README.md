## Sentiment Analysis on Amazon Electronics reviews

Provided by: Ali Mahzoon

---
### Problem
The applications of sentiment analysis are broad and powerful. The ability to extract insights from data is a practice that is being widely adopted by organizations across the world.
Being able to quickly see the sentiment behind every review means being better able to strategize and plan for the future.
The whole purpose of this analysis is being able to detect negative reviews. Detecting negative reviews is beneficial, by allowing the manufacturer to reach their customers and satisfy them in the best possible way, as well as create a win-win situation, in which the customer benefits from an improved product and the manufacturer may keep its customers.
Application of automated text classification techniques using Artificial Intelligence (AI) has consistently shown higher accuracy and speed than manual classification. Hence, proposing a system performing sentiment analysis by using Deep Learning Algorithms such as Artificial Neural Network (ANN), and Recurrent Neural Network (RNN) would be helpful to companies all around the world.

---
### Questions:
1. How can we reduce inefficiencies and noise in  a network while we increase signal?
2. Which model works the best and why?
3. what work must be done to ensure better results in the future?

---
### Dataset and Approach
* Dataset and Tools:
   * Fetched the data from Amazon Simple Storage Service (aka. Amazon s3 bucket)
   * Tensorflow, Keras, nltk, and scikit-learn library
   * Google colab
   * handled curse of dimensionality by using TSNE
   * Designed Feed Forward Neural Networks and Recurrent Neural Networks
   * Long Short Term Memory (LSTM) layer and word Embeddings by using word2vec
   * Natural Language Processing (NLP)


* Approach:
  * Create a binary classification model to classify reviews into two classes.
    1.  Not_Negative
    2.  Negative


  * Model selected based on the improvement of accuracy and efficiency
  * Designed Network improved by reducing inefficiencies and noise
  * Designed a ratio to help the network on the right path
  * Used holdout dataset and earlystopping to prevent overfitting
  * Data visualized by Bokeh, Matplotlib, seaborn and IPython

  ---
### Findings
In this dataset there are 3,091,024 reviews in 5 classes based on the star ratings that customers have given their purchased product. As we can see in the following picture, after cleaning and subsampling, we are dealing with 1,012,976 reviews. I consider the star rating 1 and 2 as Negative connotation reviews and 3, 4, and 5 as Not_Negative Reviews.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/1.png "Pre EDA")

I designed a Feed Forward Neural Network that has 142,478 units in the input layer, 10 units in the hidden layer, and a sigmoid function for output layer, forward propagation, and backward propagation. This Network gets a review, tokenize it and does the count vectorization for each review and calculates the probability of each review being Negative or Not_negative. The following image shows the network struggling to improve its accuracy.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/2.png "Feed Forward Neural Network")

We know forward propagation is a weighted sum (actually weighted sum of weighted sums), how high the number of a neuron is affects how dominantly these weights control the hidden layer and these weights control how dominantly this input affects the hidden layer (funny that they are interpreted by each other!). Most of the words with higher frequencies are irrelevant words and have nothing to do with sentiment and their weighting is casing it to have a dominant effect on the hidden layer and hidden layer is all that output layer gets to use to make a prediction, so the hidden layer is not going to have rich information. Maybe counting words is not a good idea, because the counts don't highlight the signal (when we look at the counts, it seems like we are looking at the noise). Therefore my idea is to instead of counts just represent each existing word in a review as 1 (not incrementing it).

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/3.png "After Noise Reduction")

* As we can see the model starts to learn much faster than before which is great!
* We eliminated a lot of our noise (by getting rid of weighted useless words)
* The model can find correlation between words so much faster (Accuracy of 80% in less than 1% of the whole data)
* We didn't eliminate stopping words, but the interesting point is that our model is capable to find the words that matter the most

---
#### Analyzing Inefficiencies in our Network
we are creating a really big vector for layer_0 (1 x 142478) and only a small proportion of it would be 1 and a large number of them would be 0 in each review. zero times anything is still a zero, all these zeros aren't doing anything. That to me is the biggest source of inefficiencies in this network. On the other hand, one times anything is just itself. so, these 1 times weights are kind of a waste. As we can see the speed of the Network is much higher now. 765 reviews per second for training and 6576 reviews per second in testing (absolutely screaming).

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/4.png "Analyzing Inefficiencies in our Network")

What can be done to make the network really cut through the obvious stuff and focus on more complex situations? we can frame the problem so, the Network can be as successful as possible by using a word polarity ratio.
We can tell the Network among all the unique words in the whole corpus which ones have negative polarity and which ones have positive polarity.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/5.png "Polarity ratio Created")

As we can see, we might lose some speed in our Testing method (because of adding word polarity computations and the minimum frequency of a word to include in the unique words), but we can make sure that the Network is on the right path to deal with unseen data.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/6.png "Test the model")

---
As we can see for this particular sentence "this is a good product" the network predicted that there is 75% chance this review is not negative and there is 25% chance this review is negative.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/7.png "Model Confidence")

However, our Network is going to be fooled by using "not" (for negation) "this is not a good product" the Network thinks that the word "good" has higher power than other words and predicts there is 58% chance that this sentence is not negative and 42% for being negative.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/8.png "Model Confidence")

---
This prediction leads us to use RNNs with a layer of LSTM (Long Short Term Memory) fed by word Embeddings:

 ![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/9.png "Model Confidence")


As we can see for this particular sentence "this is a good product" the network predicted that there is 99% chance this review is not negative and there is 1% chance this review is negative.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/10.png "Model Confidence")


And also our Network is successfully predicted the negated sentence "this is not a good product" there is 8% chance this review is not negative and there is 92% chance this review is negative.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/11.png "Model Confidence")

---
### Future work
Do Sentiment analysis based on a specific product or a brand. It helps the selling company (Amazon, in this case) to check the worthiness of a product or a brand and if it is profitable for them to stock it in warehouses or not.

Compare reviews of a product in other electronics stores like Best-Buy to see how people respond to a product in different stores. It can also help the manufacturer to analyze its products in online or in-person selling.

---
### Conclusion
By using an RNN Network our accuracy increased (from 83% to 87%). Moreover, our model now has the ability to understand the negation with more confidence.

![Image](https://github.com/alimahzoon/Sentiment-Analysis-/blob/main/Images/12.png "Evaluation Metrics of the best Model")

---
### Recommendations


After taking a quick look at misclassified data, I realized a considerable amount of negative reviews are rated with 5 stars or positive reviews are rated with 1 star by users mistakenly.

 This can lead to mistraining for a model. Although we might have 87% overall accuracy for prediction, I believe it should be around 90% because of human error.

To sum up, using a large number of reviews (in this case +1 million reviews) can help the model to cut through data and predict what is actually right!
