import invoke

@invoke.task
def inspect_signal_reconstruction(c):
    c.run("jupyter execute notebooks/inspect_brainlm_data_processing.ipynb")


