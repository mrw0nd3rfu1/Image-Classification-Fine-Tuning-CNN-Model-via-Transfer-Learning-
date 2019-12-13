The module to be installed is keras which is necessary for the execution of the project.
I created a ipython file but failed to make it work on it as it requires keras to function.
I will include both my python file and ipython notebook file for the review in the code folder.

The Capstone Proposal review link is-
https://review.udacity.com/#!/reviews/1251377
You can download the project from this link-
https://review-api.udacity.com/api/v1/submissions/1251377/archive

As for execution-
1. Download the dataset from https://my.pcloud.com/publink/show?code=VZvzMlZyPYO1hSn92LXdzmGNr9y1j7qKDzX
2. The model uploaded is a trained one we can alse re-train it.
3. To train the data just open the fine_tune.py and run it using the commands:
	python code/fine_tune.py <data_dir/> <model_dir/>
4. After Training classify the data model we have trained and make changes in it according to us:
	It can be executed as:
	python code/classify.py <model_dir/> <test_dir/> <results_dir/>
5. The model test and data are in the above links and result can be found by executing the data.