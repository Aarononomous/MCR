# mcr

__mcr__ is a Python module for working on the Minimum Cover Removal problem ([Erickson and LaValle, 2013](https://www.semanticscholar.org/paper/A-Simple-but-NP-Hard-Motion-Planning-Problem-Erickson-LaValle/0a9a3a6249eea0cf31646a1c97c822c0213381b7)). __mcr__ contains functions for creating these from scratch or by svg import. Once created, use __mcr__ methods to create graphs from the overlaps.

## Getting Started

### Prerequisites

__mcr__ (Python 3.6) has been written for use in Jupyter, and depends not only on the __jupyter__ module, but also on __shapely__, __graphviz__, and __matplotlib__.

I've found it easiest to deal with the dependencies by setting up a separate conda env, then `pip install XXX` for all of these, using the conda `pip`, not the default one.

### Installation

1. Download the entire project into your working directory, by direct download, cloning, etc.

2. Start a new Jupyter session: `> jupyter notebook`.

3. Inside the notebook: `include mcr`, or `from mcr import *`, etc.

4. That's it!

## Development and Contributing

This is under heavy development. Please don't let that dissuade you from sending suggestions or requests.

The Jupyter notebook within this module is used for testing. If you change the code in the Python files, don't forget to `mcr.reload()`!

### TODO

1. split MCR into files

2. remove space in labels

1. clip field to [0,1]

1. add nodes for empty spaces

1. add graph functions (as stubs)to MCR

1. display graphs over the image

1. implement basic graph-handling

1. import shape colors from SVG's

1. overlap colors?

1. graph library

1.add tolerance for point comparisons

1. shape adjacency to (labeled) graph - partition of the plane
    * special case where $s$ and $g$ are in the same partition

1. When ordering obstacles by property x in some algorithm, can we color them to show the order?


## Author

* [Aaron Jacobson](http://aaron-jacobson.com)

## Contact

Please contact by [email](mailto:hi@aaron-jacobson.com) or via the "pull request" button above.

