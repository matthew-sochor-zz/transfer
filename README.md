#Transfer - making transfer learning easy

This is a command line tool to perform transfer learning of image classification.

Currently, it will re-learn the resnet50 model trained on ImageNet.

## Setup

Prior to starting, organize the pictures you want to classify by label in a folder.

In a theoretical example where you are classifying if something is a hat or a donkey:

'''
~/donkey-vs-hat/img_data/hat/hat_1.jpg
~/donkey-vs-hat/img_data/hat/hat_2.jpg
...
~/donkey-vs-hat/img_data/donkey/donkey_1.jpg
~/donkey-vs-hat/img_data/donkey/super_cute_donkey.jpg
~/donkey-vs-hat/img_data/donkey/donkey_in_a_hat_this_is_going_to_be_bad.jpg
...
'''

Basically put all of your hat pictures in:

`~/donkey-vs-hat/img_data/hat`

and all of your donkey pictures in:

`~/donkey-vs-hat/img_data/donkey`

## Classifying images with transfer.

First configure a project with:

`transfer --configure`

or

`transfer -c`

Follow the prompts to point to your parent image directory (~/donkey-vs-hat/img_data in the above example) and to provide modeling parameters.

Run your project with:

`transfer --run`

or 

`transfer -r`

## Re-fine your model.

Want to re-fine your model with subsequent runs?  Go ahead by simply running again:

`transfer -r`

## Output

All of your generated data and model weights are saved alongside the starting images.

In the above example:

`~/donkey_vs_hat/`

## Predict on directories or files

You can either predict on your best model or last model.  To predict on your default image set, simply type:

`transfer --best-predict`

Or specify a path (can be a single file or a directory!):

`transfer --best-predict ~/path/to/your/files`

