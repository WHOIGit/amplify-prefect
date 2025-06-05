from prefect import flow

my_flow = flow.from_source(
                source="https://github.com/WHOIGit/amplify-prefect.git",
                entrypoint="src/infer_yolo_workflow.py:yolo_infer"
                )


if __name__ == "__main__":
        my_flow.serve(name="yolo-inference")
