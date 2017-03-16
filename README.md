# mcr

__mcr__ is a Python module for working on the Minimum Constraint Removal problem ([Erickson and LaValle, 2013](https://www.semanticscholar.org/paper/A-Simple-but-NP-Hard-Motion-Planning-Problem-Erickson-LaValle/0a9a3a6249eea0cf31646a1c97c822c0213381b7)). __mcr__ contains functions for creating these from scratch or by svg import. Once created, use __mcr__ methods to create graphs from the overlaps.

## Getting Started

### Prerequisites

__mcr__ (Python 3.6) has been written for use in Jupyter, and depends not only on the __jupyter__ module, but also on __shapely__, __graphviz__, and __matplotlib__.

I've found it easiest to deal with the dependencies by setting up a separate conda env, then `pip install XXX` for all of these, using the conda `pip`, not the default one.

### Installation

1. Download the entire project into your working directory, by direct download, cloning, etc.

2. Start a new Jupyter session: `> jupyter notebook`.

3. Inside the notebook: `include mcr`, or `from mcr import *`, etc.

4. That's it!

### Usage

See the Jupyter notebook [`mcr.ipynb`](mcr.ipynb) for how this is used in practice.

__NEW!:__ Displaying the graph isn't automatic anymore. See the `show()` method or the notebook for more details. Basically, all of the methods that previously displayed the field now only draw to it, and displaying the finished product (from all of the many (??) additions is up to you.) `show()` is the only method on `MCR` that actually calls `plt.show()`.

__`show()`:__

1. `self.plot_obstacles()`

1. `self.plot_graph()`

1. `MCR.setup_axes()`

1. `plt.show()`

## Development and Contributing

This is under heavy development. Please don't let that dissuade you from sending suggestions, requests, or especially pull requests.

The Jupyter notebook within this module is used for testing. If you change the code in the Python files, don't forget to `importtools.reload(mcr)` or restart the Jupyter kernel!

__N.b.:__ If you're submitting a change to `mcr.ipynb`, the Jupyter notebook, *please* clear output from the cells beforehand. This can be done by __Cell > All Output > Clear__ or __Kernel > Restart and Clear Output__ in the Jupyter menubar.

### TODO

1. Need a way to easily highlight featured obstacles, vertices, and edges
    - label certain vertices
    - mark certain edges
    - highlight certain faces
    - highlight certain obstacles

1. From a subgraph, need to pass along the labels and positions associated with the nodes
    - if the subgraph has edges w/o nodes, still need the labels and locations of the induced nodes
    - I think this is done for us, I just need to show it

1. When ordering obstacles by property x in some algorithm, can we color them all to show the order?

1. add tolerance for point comparisons

### Nice-to-haves

1. import shape colors from SVG's?

## Authors

* [Aaron Jacobson](http://aaron-jacobson.com)

## Contact

Please contact by [email](mailto:hi@aaron-jacobson.com) or via the "pull request" button above.

