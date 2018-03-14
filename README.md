# Transfer - making transfer learning easy

This is a command line tool to perform transfer learning of image classification.

Currently, it will re-learn the resnet50, inception_v3 or xception models pre-trained on ImageNet.

Furthermore, it provides a framework to serve your models!  You can export a trained model, import it on another computer later, make local predictions or setup a rest api to make predictions.

[Here are some models I have trained ready for prediction!](http://www.mattso.ch/transfer-models)

## How is this different?  Why transfer?

Transfer pre-calculates and saves the early layer outputs for each model.  It then re-learns the final several layers and just those layers.  This differs from other transfer learning approaches that only learn the very final layer and then relearns every layer in that it is *faster* because it calculates the early layers once and *still very accurate*.  It also has built in support for some machine learning best practices like k-fold validation and ensembling of k-fold models.

As a benchmark, using transfer I was able to score a 0.96599 on the [plant seedling classification](https://www.kaggle.com/c/plant-seedlings-classification) competition on Kaggle.  Not quite as good as the fastai benchmark of ~0.98, but good enough for many applications!

Finally, transfer is meant to be a model delivery platform as well.  Train a model with transfer, export it to save it, re-import it elsewhere via transfer, and make predictions on new images!  Its a great way to share models with friends, colleagues and collaborators.

Transfer can manage multiple models simultaneously via a simple project based organization.

## What is the community saying
- **@thenomemac : You could probably code something better, but why... Just use transfer**
- **@anonymous : Even I can use this!**

## Software to pre-install

Please first install [tensorflow](https://www.tensorflow.org/install/) and python 3.  I recommend installing the latest python via [Anaconda](https://anaconda.org/anaconda/python).

Install transfer with

`pip install transfer`

Thats it!  You can test that transfer is correctly installed by typing:

`transfer`

You should see help and a list of available commands.  Now we just need some images to classify.

## Get your images ready!

Prior to starting, organize the pictures you want to classify by label in a folder.  A great example of a project already organized like this is the Kaggle competition for [plant seedling classification](https://www.kaggle.com/c/plant-seedlings-classification).

In a theoretical example where you are classifying if something is a **hat** or a **donkey** you would organize the images in the following manner:

```
~/donkey-vs-hat/hat/hat_1.jpg
~/donkey-vs-hat/hat/hat_2.jpg
~/donkey-vs-hat/hat/ridiculous_proper_english_lady_hat.jpg
...
~/donkey-vs-hat/donkey/donkey_1.jpg
~/donkey-vs-hat/donkey/super_cute_donkey.jpg
~/donkey-vs-hat/donkey/donkey_in_tree.jpg
...
```

Basically put all of your hat pictures in:

`~/donkey-vs-hat/hat`

and all of your donkey pictures in:

`~/donkey-vs-hat/donkey`

## Classifying images with transfer

First configure a project with:

`transfer --configure`

Follow the prompts to point to your parent image directory (`~/donkey-vs-hat` in the above example) and to provide modeling parameters.

You can always see your projects by inspecting the local configuration file:

`~/.transfer/config.yaml`

## Train your models!

Train your model with:

`transfer --run`

## Predict on an image or directory

Transfer provides two modes to predict your models with, either make local predictions on either a directory or single images with:

`transfer --predict PATH_TO_IMAGES`

or serve your model via a simple local rest-api:

`transfer --prediction-rest-api`

## Save and share your model

Great, so you trained a model and you can make predictions.  Now what?  You can save your model and configuration for later import on another computer (with transfer installed, obviously) or even give it to a friend (they probably have difficulty telling the difference between donkeys and hats?)

Export your model with:

`transfer --export`

## Import pre-trained project

Did your friend send you a donkey-vs-hat model trained with transfer?  Well, how about we import that:

`transfer --import IMPORT_CONFIG`

where IMPORT_CONFIG is the path to tar.gz file where the config.yaml and model files are.

[Here are some models I have trained ready for prediction!](http://www.mattso.ch/transfer-models)

## Contribute

Please, if you use transfer and run into any issues or have suggestions for new features, submit an issue on Github or make a pull request.
