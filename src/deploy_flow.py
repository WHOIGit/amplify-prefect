import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from prefect import flow

# Site settings for flow runs (e.g. AMPLIFY_DOJO_IMAGE), kept out of the repo.
# Flow runs inherit this process's environment.
FLOWS_ENV = Path(os.environ.get("AMPLIFY_FLOWS_ENV", Path.home() / ".config/amplify-prefect/flows.env"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a Prefect flow from Git storage.")
    parser.add_argument("source", type=str, help="Git repository where flow code is stored.")
    parser.add_argument("entrypoint", type=str, help="File and function containing flow code.")
    parser.add_argument("name", type=str, help="Name of the flow to be displayed in the Prefect UI.")
    args = parser.parse_args()
    load_dotenv(FLOWS_ENV)

    flow_deployment = flow.from_source(
            source=args.source,
            entrypoint=args.entrypoint
            )
    flow_deployment.serve(name=args.name)
