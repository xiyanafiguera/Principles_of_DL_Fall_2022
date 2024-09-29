import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import argparse


def execute_notebook(notebook_filename):
    with open(notebook_filename) as f:
        nb_in = nbformat.read(f, nbformat.NO_CONVERT)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    nb_out = ep.preprocess(nb_in)

    return nb_out 

def test_gpu(question):
    nb_out = execute_notebook('Multi_GPU_assignment.ipynb')
    q1 = nb_out[0]['cells'][-2]['outputs'][-1]['text'].strip().split('\n')[0]
    q2 = nb_out[0]['cells'][-2]['outputs'][-1]['text'].strip().split('\n')[1]
    q3 = nb_out[0]['cells'][-2]['outputs'][-1]['text'].strip().split('\n')[2]
    q4 = nb_out[0]['cells'][-2]['outputs'][-1]['text'].strip().split('\n')[3]
    q5_1 = nb_out[0]['cells'][-2]['outputs'][-1]['text'].strip().split('\n')[4]
    q5_2 = nb_out[0]['cells'][-2]['outputs'][-1]['text'].strip().split('\n')[5]

    if question == 1:
        assert q1 == 'False', "Q1: DP can only trained on single machine, while DDP can trained on both single and multiple machines."
    
    elif question == 2:
        assert q2 == 'True', "Q2: DP collects all the outputs in a single GPU that causes imbalanced memory usage."

    elif question == 3:
        assert q3 == 'False', "Q3: DDP computes loss independently after loss on each GPU, while DP collects on a single GPU to compute a loss."

    elif question == 4:
        assert q4 == '[0, 2, 4]', "Q4: First, thrid, and fifth GPU ids are 0, 2, and 4." 

    elif question == 5:
        assert 'nnodes=1' in q5_1 or 'standalone' in q5_1, "Q5_1: The correct answer is either 'nnodes=1' or 'standalone'."

    elif question == 6:
        assert'nproc_per_node=6' in q5_2 or '6' in q5_2, "Q5_2: The correct answer is either 'nproc_per_node=6' or '6'."

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the notebook')
    parser.add_argument('--question', type=int)
    args = parser.parse_args()
    test_gpu(args.question)