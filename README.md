# wslda-vortex-docker

## to start the app

- in docker desktop:
  - settings -> resources -> WSL integration -> Enable Integration with my default WSL distro (windows only)
  - settings -> Features in development -> Enable Host Networking (experimental on windows)

- `cp ./env-dev.env.example ./env-dev.env`

- `npm run dev_restart_linux` or\
  `npm run dev_restart_windows`\
  is preferred over plain `docker compose up -d` because of the custom names of `docker-compose-dev.yml` and `env-dev.env` files

- `npm run dev_restart_purge_linux` or\
  `npm run dev_restart_purge_windows`\
  to reset the app to the "factory settings" configuration\
  (WARNING: by design this will NOT reverse any `git checkout`s done to the submodules !!!)

- either:
  - use VS code to attach to a running container ("ctr + shift + p" -> "Dev Containers: Attach to Running Container" -> "wslda-vortex-analysis-1")
  - `npm run dev_exec_analysis`\
    to ssh into the container via the console

- from within the container:\
  `/opt/python/3.12/venv/bin/python3 /workspace/app/src/start.py`

---

## by default:

1. projects expects data from https://arxiv.org/abs/2201.07626 to ba available under:\
   `<project's root>/results/data/st-vortex-recreation-01/`

2. output results of analysis to:\
   `<project's root>/results/analysis/`
