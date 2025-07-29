import invoke

from . import prep, notebooks, models


@invoke.task
def clean(c):
    c.run("rm -rf ./data/interim/*")

# The main namespace MUST be named `namespace` or `ns`.
# See: http://docs.pyinvoke.org/en/1.2/concepts/namespaces.html
ns = invoke.Collection()

ns.add_collection(prep, name='prepare')
ns.add_collection(notebooks)
ns.add_collection(models)
ns.add_task(clean)