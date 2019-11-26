Directory structure
--------------------------------
The dataset is partitioned into a training and test set. These sets can be
found in the directories "train" and "test", respectively. The training set can
be used in any way the user sees fit. In particular, if a separate validation
set is needed, it can be obtained as a subset of the training set.
The test set is to be used for evaluation purposes only.

The sub-directory "script" contains the official evaluation script and an
example of a detection file.


Groundtruth data format
--------------------------------
For each image there exists a text file with the extension .gt_data.txt which
contains the object annotations. 

Each line in this file corresponds to an object instance and is to be
interpreted in the following way:

<x1> <y1> <x2> <y2> <class_id> <dummy_value> <mask> <difficult> <truncated>

Since this dataset only contains annotations for a single object class,
the only class_id is always 0.

All other information can and should be disregarded for the purpose of this project.


Evaluation script
--------------------------------
We provide an evaluation script which is able to compute:

- A Precision-Recall curve for each class
- The average precision (AP) for each class
- The mean average precision (mAP)

The evaluation script is written in Python 3 and can be found in the sub-
directory "scripts". Aside from a working Python interpreter the script
relies on Numpy and Matplotlib which are libraries that might not be
installed by default in your Python distribution. All other modules should
be included in any Python distribution by default.

The script has been tested on both Linux (Ubuntu 14.04) and Windows (8.1).
Should you encounter problems on your system, please contact us.

The evaluation script can be invoked as follows:
python eval_detections.py 
  --detection    <Path to the file containing the detections>
  --dset_basedir <Path to the root directory of the dataset>
  --resfile      <(optional) File prefix of the result files to be written>
  --genplots     <(optional) Set to "true" in order to generate PR curves>
  --overwrite    <(optional) Set to "true" in order to overwrite result files
  --clip_thresh  <(optional) Exclude detections with lower score than threshold
  

Detection data format
--------------------------------
The evaluation script expects the detections in the form of a text file. 
In this file, each line represents a detection in the following format:

<Image name> <class_id> <x1> <y1> <x2> <y2> <score>

An example of such a file can be found in the folder "script".
