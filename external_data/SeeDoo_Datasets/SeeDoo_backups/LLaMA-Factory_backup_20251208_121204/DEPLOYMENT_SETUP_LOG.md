# LLaMA-Factory Deployment Log

  ## Quick Reference

  **Server:** seedoo@hetzner-gpu.tail9e6e7.ts.net
  **Project Path:** /home/seedoo/shahar_linux_wsl/LLaMA-Factory/
  **Container Name:** llamafactory
  **API Port:** 8002 (host) -> 8000 (container)
  **UI Port:** 7860

  ## Deployment Steps

  ### 1. Sync Files from Local to Server

  ```bash
  LOCAL_USER=$(whoami)
  FOLDER_NAME=$(basename "$PWD")
  rsync -avz \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='node_modules' --exclude='*.log' --exclude='output' \
    --exclude='hf_cache' --exclude='ms_cache' \
    --exclude='data/treadmill_videos' --exclude='data/*_train/' \
    --exclude='data/*_test/' --exclude='*.mp4' --exclude='*.avi' \
    "$PWD/" \
    seedoo@hetzner-gpu.tail9e6e7.ts.net:/home/seedoo/$LOCAL_USER/$FOLDER_NAME/

  What Gets Synced: Code, configs, JSON dataset definitions
  What Gets Excluded: Models, videos, generated datasets (re-downloaded/generated on server)

  2. SSH to Server

  ssh seedoo@hetzner-gpu.tail9e6e7.ts.net
  cd /home/seedoo/shahar_linux_wsl/LLaMA-Factory/

  3. Build and Start Container

  # Build image from source
  docker-compose up -d --build

  # Verify running
  docker ps | grep llamafactory

  4. Run Pipeline

  # Direct execution
  docker exec -it llamafactory python run_full_pipeline.py

  # Or enter container interactively
  docker exec -it llamafactory bash
  python run_full_pipeline.py

  Troubleshooting

  Port Conflicts

  - Port 8000 was in use, changed to 8002
  - Check free ports: sudo lsof -i :<port>

  Missing Image

  - Custom image doesn't sync via rsync
  - Solution: Build from Dockerfile on server

  YAML Errors

  - Use nano to edit docker-compose.yml directly
  - Heredoc commands can fail with complex YAML

  Important Notes

  - Models download faster on server
  - Videos generated automatically
  - All running containers are essential - don't stop them
  - GPU access configured via docker-compose
  DOCEND

  **Run the commands above to:**
  1. Find a free port
  2. Update docker-compose.yml to use port 8002
  3. Create documentation
  4. Start the container
