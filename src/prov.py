import os
from datetime import datetime, timezone
from functools import wraps

from prefect import Task
from prefect.client.schemas.objects import TaskRun
from prefect.states import State

from provenance.client import ProvenanceClient, ProvType, ProvVerb

def on_task_complete(tsk: Task, run: TaskRun, state: State) -> None:
   
    with ProvenanceClient(os.getenv("PROVENANCE_STORE_URL")) as prov_client:
        task_label = f"task-{run.task_key}"
        flow_label = f"flow-{run.flow_run_id}"

        # Create task and flow nodes
        prov_client.create_node(task_label, ProvType.ACTIVITY, description=tsk.name)
        prov_client.create_node(
            flow_label, ProvType.ACTIVITY, description=str(run.flow_run_id)
        )

        # Link task to flow
        prov_client.create_relation(
            subject_label=flow_label,
            verb=ProvVerb.WAS_GENERATED_BY,
            object_label=task_label,
            run_id=f"{run.flow_run_id}_{run.task_key}",
            start_time=datetime.now(timezone.utc),
        )
