import os

def generate_hostfile():
    num_gpus = os.getenv("WORKER_GPU", "8")
    hostfile_content = ""
    with open("/opt/tiger/hostfile", 'r') as f:
        for worker_host in f:
            worker_host = worker_host.strip()
            hostfile_content += f"{worker_host} slots={num_gpus}\n"

    return hostfile_content.strip()

hostfile_path = "deepspeed_hostfile"
with open(hostfile_path, "w") as f:
    f.write(generate_hostfile())

print(f"Hostfile content written to {hostfile_path}")