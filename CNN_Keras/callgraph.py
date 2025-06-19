from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from pycallgraph2.globbing_filter import GlobbingFilter
from pycallgraph2.config import Config
import runpy

# Define include/exclude rules
trace_filter = GlobbingFilter(
    include=['CNN_MNIST*'],
    exclude=['tensorflow.*', 'keras.*', 'wrapt.*']
)

# Create and configure the profiler config
config = Config()
config.trace_filter = trace_filter

# Set output file
graphviz = GraphvizOutput()
graphviz.output_file = 'CNN_MNIST_callgraph.png'

# Profile the script execution
with PyCallGraph(output=graphviz, config=config):
    runpy.run_path('CNN_MNIST.py', run_name='__main__')
