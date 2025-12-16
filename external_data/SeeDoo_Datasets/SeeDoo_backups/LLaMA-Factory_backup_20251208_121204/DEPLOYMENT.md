# :rocket: Universal Hetzner Deployment (Rsync-Based)

To deploy the current project to the Hetzner production server, use the command:

    deploy to hetzner

When this command is invoked, the deployment system performs a safe, file-only
synchronization using `rsync` with the following behavior:

---

## 1. Automatic Local Discovery

The system automatically detects:

- **Your local username** (e.g. "alexlan")
- **The absolute path of your current working directory**
- **The folder name of the project you are currently inside**, e.g.:

      /Users/<LOCAL_USERNAME>/.../<CURRENT_FOLDER_NAME>

This folder becomes the root of what gets deployed.

---

## 2. Destination Path on Hetzner

All deployments are synchronized to the following path:

    seedoo@hetzner-gpu.tail9e6e7.ts.net:/home/seedoo/<LOCAL_USERNAME>/<CURRENT_FOLDER_NAME>/

This ensures:

- Each developer gets isolated, predictable directories on Hetzner.
- Deploying from any folder on your machine automatically produces a matching
  folder name on the remote.
- No project-specific configuration is required.

Examples:

Local CWD:
    /Users/john/Code/VideoMonitor/

Remote target becomes:
    /home/seedoo/john/VideoMonitor/

Local CWD:
    /Users/sarah/Desktop/awesome-tool/

Remote target becomes:
    /home/seedoo/sarah/awesome-tool/

---

## 3. Rsync Behavior

Deployment is performed via an `rsync -avz` transfer with the following exclusions:

    --exclude='.git'
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='node_modules'
    --exclude='*.log'
    --exclude='output'
    --exclude='videos-resized'

This ensures that only relevant source files are transferred and no heavy,
temporary, or auto-generated directories are copied.

Rsync is recursive by default and only transfers changed files, making
deployments efficient.

---

## 4. Important Notes

- This deployment action **only synchronizes files**.  
- It performs **no Docker builds**, **no restarts**, and **no remote commands**.  
- It can safely be run repeatedly without overwriting unrelated directories.  
- It always mirrors **the folder you are currently inside**, regardless of the project.

---

Use "deploy to hetzner" whenever you want to push your current working directory
to the Hetzner server via rsync.

Any future commands we run we can run via ssh on the server, inside a container the user will define. 
via docker run or docker excec. But any container operations like up or down will require always user approval before you proceeed`

this is an example for ssh: ssh seedoo@hetzner-gpu.tail9e6e7.ts.net

and how we did rsync with out the models or videos:
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

    dont ever close or abort running containers or busy ports!!!
