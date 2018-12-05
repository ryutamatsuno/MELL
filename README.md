# MELL
MELL is short for Multiplex network Embedding via Learning Layer vectors. This method is for embedding multiplex network. Multiplex network is a multi-layer network where all layers have the same set of nodes.

PDF: https://dl.acm.org/citation.cfm?id=3191565

# How to use

Firstly, this program requires python 3, tensorflow and numpy libraries.
We use python 3.6.3, tensorflow 1.1.0, and numpy 1.14.1.

To test MELL, you can run a following command.

```shell
python main.py
```

This command uses the sample data set in "Dataset" folder.
we provide two data set, and you can use these for trying.

You also can use MELL.py directly for your own experiment. Then the codes will be
like following codes.

```python
from MELL.MELL import MELL_model


# you should get L, N, directed, edges for training and edges for testing from your data set
# you also should decide the hyper parameters: d, k, lamm, beta, gamma

# define the model
model = MELL_model(L, N, directed, train_edges, d, k, lamm, beta, gamma)
# train
model.train(500)
# predict
prediction = [ model.predict(t) for t in test_edges]
```

# License

Creative Commons Attribution 4.0 International (CC BY 4.0) License.

