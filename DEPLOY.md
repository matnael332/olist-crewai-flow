# Deploying pulse on our server

pulse runs on one Linux server we control, with Docker Compose. This guide takes a fresh server
to a running dashboard and API on the synthetic demo customer. The CI `docker` job runs the same
commands on every push, so they are known to work.

## What runs where

| Piece | What it is |
|---|---|
| `pulse:local` image | the whole product: pipeline, dashboard, API (`Dockerfile`) |
| `dashboard` service | Streamlit on port 8501, always up |
| `api` service | prediction API on port 8000, always up |
| `job` service | runs a command on demand (`synth`, the pipeline); not started by `up` |
| `pulse-data` volume | all customer data: `/srv/pulse/customers/<slug>/{data,runs,artifacts}` |

The image holds code only. Customer data, runs and models live in the volume, so rebuilding or
updating the image never touches them. `PULSE_CUSTOMER` (in `.env`, default `demo`) picks the
customer folder the dashboard and the API show.

**Both ports listen on the server's localhost only.** There is no login yet (that is step 6), so
nothing is exposed to the internet; you open the dashboard through an SSH tunnel.

## 1. The server

- Ubuntu 24.04, 4 vCPU, 8 GB RAM, 80 GB disk (e.g. Hetzner CX32 or a DigitalOcean droplet of
  the same size).
- An SSH key at creation; no password login.
- Firewall: allow SSH only.

```bash
sudo ufw allow OpenSSH && sudo ufw enable
```

Sizing: the synthetic demo (200 accounts) is about 145 MB of raw tables and a full run takes a
few minutes. Revisit the size once the real EZtime export size is known.

## 2. Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"     # log out and back in once
docker compose version              # v2.24 or newer
```

## 3. The code

```bash
git clone <repository url> pulse && cd pulse
cp .env.example .env                # optional: GEMINI_API_KEY for LLM mode, PULSE_CUSTOMER
docker compose build                # first build takes a few minutes
```

## 4. The demo customer

```bash
C=/srv/pulse/customers/demo
docker compose run --rm job synth --data-dir $C/data
docker compose run --rm job python -m pulse_flow.run --no-llm \
  --spec pulse_specs.workforce_saas.spec:SPEC \
  --data-dir $C/data --runs-dir $C/runs --artifacts-dir $C/artifacts
docker compose up -d
docker compose ps                   # dashboard and api: healthy
```

`$C` expands on the server, and the path is the same inside the container (the volume is
mounted at `/srv/pulse`). Exit code 0 means the run passed the three gates and was promoted to
`$C/artifacts`; 1 means a graceful failure (`$C/runs/<id>/run_status.json` says why).

## 5. Open the dashboard

From your own computer:

```bash
ssh -L 8501:localhost:8501 -L 8000:localhost:8000 <user>@<server>
```

Then browse to http://localhost:8501 (dashboard) and http://localhost:8000/health (API).

## Day to day

| Task | Command |
|---|---|
| New run for a customer | the `pulse_flow.run` command above with that customer's `C`, then `docker compose restart api` (the API loads the model at start; the dashboard picks up new files by itself) |
| Show another customer | set `PULSE_CUSTOMER=<slug>` in `.env`, then `docker compose up -d` |
| Update to a new version | `git pull && docker compose up -d --build` |
| Logs | `docker compose logs -f dashboard api` |
| Back up all customer data | `docker run --rm -v pulse_pulse-data:/srv/pulse -v "$PWD":/backup busybox tar czf /backup/pulse-data-$(date +%F).tgz -C /srv pulse` |
| Stop | `docker compose down` (the volume stays; `down -v` deletes it with all customer data) |

The volume name is `<folder>_pulse-data`, so `pulse_pulse-data` when the repository folder is
called `pulse` (`docker volume ls` shows it).

## Not yet (step 6)

HTTPS and a domain, a login per customer, uploading a customer's export, and a "run" button.
Until then customer data is copied into the volume by hand and runs are started from the shell.
