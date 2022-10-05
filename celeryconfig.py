broker_url = "redis://localhost:6379"
result_backend = "redis://localhost:6379"

task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
# timezone = "Europe/Oslo"
enable_utc = True

# Route model-specific tasks to model-specific queues
task_routes = {
    "metaseq.cli.interactive_hosted.compute_gptz": {"queue": "gptz"},
    "tasks.add": {"queue": "sample-queue"},
}

task_annotations = {"tasks.add": {"rate_limit": "10/m"}}

# Ensure each worker handles only 1 request at a time
task_acks_late = True
# worker_prefetch_multiplier = 1
