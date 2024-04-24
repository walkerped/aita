# AITA Judge Bot FAQ

1. What is Judge Bot?

    [Judge Bot](https://twitter.com/AITA_judgebot) is program that makes predictions about the outcomes of posts on the [AITA (Am I the Asshole) subreddit](https://www.reddit.com/r/AmItheAsshole/) and tweets these predictions. These predictions are based on a neural network that has been trained to do this task.

2. What is the AITA subreddit? What exactly is Judge Bot predicting?

    AITA is a reddit forum (AKA a subreddit) where people describe a situation that happened to them, and ask if they were in the wrong. Others can comment on these posts and vote about whether the original poster was in the wrong.  The main vote options are YTA (you're the asshole) and NTA (not the asshole). The comment that gets the most votes determines whether the original poster is deemed to be an asshole or not.

    Judge bot finds new posts, and predicts the outcome of the reddit vote. Judge bot tweets that start with "PREDICTION UPDATE:" are tweets where Judge bot assesses whether it was correct on a prior prediction, after the reddit vote is in.

3. How does Judge bot make its predictions?

    Judge bot is a fine tuning of [BigBird](https://huggingface.co/docs/transformers/en/model_doc/big_bird), a sparse attention transformer model that is specialized for long texts. I chose BigBird because AITA posts are typically several paragraphs long, and often reading a single paragraph doesn't give enough context to make a judgement - you really need to digest the whole thing.

    BigBird was trained to predict AITA post outcomes on 130,000 posts.  The training dataset had an equal number of YTA and NTA posts. Judgebot's neural network model was finalized after many round of hyperparameter tuning, validated on 10,000 posts held out from the training set, and tested on 10,000 more posts.

4. How accurate is Judge bot?

    Judge bot's neural network was tested on 10,000 posts that were indepedent from those used in training and validation and had an equal number of YTA and NTA outcomes. It was 71% accurate.

5. So Judge bot was 71% accurate when tested. Is that good?

    There's no concrete metric for what accuracy would be considered good. Since Judge Bot was tested on a dataset with an equal number of YTA and NTA posts, we would expect 50% accuracy if Judge Bot hadn't learned anything and was basically 'guessing'. When I tested myself on this task (on 25 posts) I was 84% accurate, so Judge Bot is better than guessing, but not at human level accuracy yet.

    As a final point of reference, I tested chatgpt 3.5 on 25 posts, after explaining the task to it.  While it provided plausible reasons for its predictions, it scored 52% - not reliably better than chance. This suggests that fine tuning a neural network to do this task (i.e. Judge Bot) did lead to better performance than using a general-purpose LLM.

6. When I look through the prediction updates on twitter, Judge Bot seems to have much better accuracy than 71%. What gives?

    When making predictions, Judge Bot's neural network produces a continuous value (i.e. logits), that can then be converted into a categorical prediction with low values being a NTA prediction, and high values being a YTA. These continuous values can be used as indicators of the model's confidence, with very high or very low values indicating a prediction the model is very confident in.

    I tested Judgebot's accuracy for the top 10% of posts Judge Bot was most confident in predicting. Among these posts, Judge Bot was 93% accurate! The code that posts predictions to twitter runs a batch of recent posts through the model, and selects predictions to tweet out that are among the top 10% for confidence. This should result in much greater accuracy among predictions that get tweeted out.

7. Isn't cherry picking the 'high confidence' predictions to be tweeted cheating?

    That's not how I would frame it! There are a lot scenarios where cherry picking the best predictions for use is a valid approach. For example, if you are using a neural network to pick stocks that will increase in price, you probably want to act on only the highest confidence predictions. For other applications (ex. identifying whether a widget is good or faulty), you may be required to generate and act on predictions for each case.

    In the case of Judge Bot, I think it is fair to select high confidence posts, in order to demonstrate Judge Bot's legal prowess. 

8. Why have you made Judge Bot?

    I've been learning about creating neural networks, and wanted to build one to crystalize what I have learned.  When brainstorming projects, this stood out for a couple reasons.  
    
    For starters, I just really like the AITA subreddit.  It is fun to see what scenarios people want to be judged on, and even more fun to see how others respond to them.

    Second, the AITA subreddit represents a source of data with a large number of posts already classified into categories. Typically to do deep learning well, you need a lot of cases. With the AITA subreddit, I have over 100,000 labeled cases that (as far as I can tell) had never been used to make a neural network before. I thought this was a great dataset to use for training a model.

    Finally, I turned the model into a twitter bot to practice putting a neural network into production, and to have an easy way to show off the model I created.

9. How does Judge Bot get run to make automated tweets everyday?

    [A Kaggle notebook](https://www.kaggle.com/code/walkerped/run-aita/) that is scheduled to run daily runs Judge Bot. This notebook downloads the git repo each day to make sure it is current, and runs predict.py (which makes prediction tweets), assess.py (prediction update tweets) and monitor.py (which saves reddit data daily for future use). This notebook uses excel files in a separate git repo to keep track of posts and predictions. This isn't really the main use case for Kaggle notebooks, so it's maybe a little klunky - but it's free, which was my main consideration.
