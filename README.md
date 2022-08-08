### Question Answering with Additive Restrictive Training -- QuAART

This repository provides code for data management and Fine Grained Entity Recognition and Typing experiments described in "Question Answering with Additive Restrictive Training (QuAART): Question Answering for the Rapid Development of New Knowledge Extraction Pipelines", to be presented at EKAW 2022, the 23rd International Conference on Knowledge Engineering and Knowledge Management. (https://ekaw2022.inf.unibz.it/accepted-papers/)

Code is designed to provide generally comparable results to those described in the paper through execution of a repeatable pipeline. Results will not be identical, as precise reproducibility is not the goal.


#### Installation

Code runs best on a GPU box and requires at least 350GB of available storage for building datasets and trainng models.

We are using Python 3, Pandas, SpaCY, and SeqEval in our evaluation scripts, and sklearn and gdown in the collection and processing of training data. We presently are using very specific versions of Huggingface Transformers and Datasets, as well as PyTorch. These are documented in the requirements.txt file.


To install necessary libraries, set up a virtual environment of your choice and run `pip install -r requirements.txt` from the root of your QuAART clone.

#### Quick Start

Once a suitable virtual environment is set up and configured per the provided requirements.txt file, executing the experiments is as simple as running `./end-to-end.sh`.

The end-to-end script takes one optional argument, which is the filename prefix for final output of evaluation csv files. It then calls 3 additional scripts:
* ./data-prep-script.sh
* ./experiment1.sh hands_$1
* ./experiment2.sh figer_$1

By default, the end-to-end script uses "results" as the filename prefix. Note that if you are re-running either experiment script, or the evaluation scripts therein, you must either use a new filename prefix, or delete the previous copies of hands_[filename].csv and figer_[filename].csv, or the script will continue appending to them and will throw off the final results.

The data prep script which downloads data from both FIGER and HAnDS repositories and a google drive folder and downloads and patches executables for Huggingface's Question Answering example. It then generates the various training and evaluation data files used in experiments 1 and 2.

The two experiment scripts then fine-tune transformers QA models for the two experiments described in the QuAART paper, run predictions, evaluate the results, and run our threshold selection algorithm.

Each of these three scripts can also be run standalone if you do not wish to run the end-to-end script.

The end-to-end script takes approximately 24 hours to run on an AWS EC2 p3.2xlarge environment, and requires at least 350GB of free space.

Special thanks to Curt Kohler and Tony Scerri for various discussions and review of this codebase.
