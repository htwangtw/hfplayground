import invoke

from . import prep


@invoke.task
def clean(c):
    c.run("rm -rf ./data/interim/*")

# The main namespace MUST be named `namespace` or `ns`.
# See: http://docs.pyinvoke.org/en/1.2/concepts/namespaces.html
ns = invoke.Collection()

prepare = invoke.Collection("prepare")
prepare.add_task(prep.data)
prepare.add_task(prep.models)
prepare.add_task(prep.atlas)
prepare.add_task(prep.brainlm_workflow_timeseries)
prepare.add_task(prep.gigaconnectome_workflow_timeseries)

ns.add_collection(prepare)
ns.add_task(clean)