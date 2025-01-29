./run_prefect.sh &
# Wait for server to start, then serve flow
sleep 10; python workflow.py