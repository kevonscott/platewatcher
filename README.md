# PlateWatcher - Lightweight LLM License Plate Extractor

Watches a directory for motion-captured image events, uses a local vision model
to detect vehicles and infer license plates, and stores results in a
local SQLite database.

---

## Prerequisites

### 1. Ollama + Moondream

```bash
# Install Ollama
# See https://ollama.com/download for more details
curl -fsSL https://ollama.com/install.sh | sh

# Pull the moondream model (lightest vision model, best for Pi)
ollama pull moondream

# Verify it's running
ollama list
```

### 2. Create a virtual environment and install PlateWatcher

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install https://github.com/kevonscott/platewatcher
```

---

## Event Directory Format

Your camera/motion software should write frames into subdirectories of
`events/`, named by the event timestamp:

```
events/
  2024-11-15_14-23-05/     ← one subdirectory per motion event
    frame_001.jpg
    frame_002.jpg
    frame_003.jpg
  2024-11-15_14-31-12/
    frame_001.jpg
    ...
```

The directory name becomes the `event_id` and `event_timestamp` in the DB, so use a format that sorts chronologically (ISO-8601 recommended).

---

## Running

```bash
.venv/bin/platewatcher
```

The process runs forever.

---

## Configuration

Use environment variables to override settings (recommended), instead of editing source files directly.

Example:

```bash
export PLATEWATCHER_APP_DIR="$HOME/.local/state/platewatcher"
export PLATEWATCHER_EVENTS_DIR="$HOME/.local/state/platewatcher/events"
export PLATEWATCHER_DB_URL="sqlite:///$HOME/.local/state/platewatcher/license_plates.db"
export PLATEWATCHER_LLM_TIMEOUT_SECONDS="90"
export PLATEWATCHER_POLL_INTERVAL_SECONDS="10"
export PLATEWATCHER_MAX_IMAGE_WIDTH="1280"
export PLATEWATCHER_MAX_IMAGE_HEIGHT="720"
export PLATEWATCHER_MAX_IMAGES_PER_EVENT="10"
export LLM_BASE_URL="http://localhost:11434"
export LLM_MODEL="moondream"
```

| Setting | Default | Env Variable | Description |
|---|---|---|---|
| `app_dir` | `~/.local/state/platewatcher` | `PLATEWATCHER_APP_DIR` | Base working directory for app state |
| `events_dir` | `~/.local/state/platewatcher/events` | `PLATEWATCHER_EVENTS_DIR` | Where camera software writes event folders |
| `db_url` | `sqlite:///$HOME/.local/state/platewatcher/license_plates.db` | `PLATEWATCHER_DB_URL` | Database URL |
| `log_file` | `~/.local/state/platewatcher/platewatcher.log` | `PLATEWATCHER_LOG_FILE` | Log file path when file logging is enabled |
| `llm_base_url` | `http://localhost:11434` | `LLM_BASE_URL` | LLM server URL |
| `vision_model` | `moondream` | `LLM_MODEL` | LLM model to use |
| `llm_timeout_seconds` | `90` | `PLATEWATCHER_LLM_TIMEOUT_SECONDS` | Per-request timeout (Pi can be slow) |
| `poll_interval_seconds` | `10` | `PLATEWATCHER_POLL_INTERVAL_SECONDS` | How often to check for new events |
| `max_image_width` | `1280` | `PLATEWATCHER_MAX_IMAGE_WIDTH` | Max image width before LLM upload |
| `max_image_height` | `720` | `PLATEWATCHER_MAX_IMAGE_HEIGHT` | Max image height before LLM upload |
| `max_images_per_event` | `10` | `PLATEWATCHER_MAX_IMAGES_PER_EVENT` | Safety cap on frames per event |

---

## Database Schema

```sql
SELECT * FROM detections;
```

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Auto Primary Key |
| `event_id` | TEXT | Unique folder name |
| `event_timestamp` | TEXT | Same as event_id (ISO-8601) |
| `processed_at` | TEXT | When processing completed |
| `status` | TEXT | `found` or `unresolved` |
| `plate_text` | TEXT | Cleaned alphanumeric plate, NULL if unresolved |
| `vehicle_color` | TEXT | e.g. `silver` |
| `vehicle_type` | TEXT | e.g. `sedan`, `truck`, `SUV` |
| `confidence` | TEXT | `high`, `medium`, or `low` |
| `image_count` | INTEGER | Frames processed |
| `notes` | TEXT | Reason for unresolved events |

### Useful queries

If you are using the default SQLite setup, open the database with:

```bash
.venv/bin/python -m sqlite3 "$HOME/.local/state/platewatcher/license_plates.db"
```
If you changed the DB path via `PLATEWATCHER_DB_URL`, use that SQLite file path instead.

Then run queries directly:

```sql
-- All successful plate reads today
SELECT event_timestamp, plate_text, vehicle_color, vehicle_type, confidence
FROM detections
WHERE status = 'found'
ORDER BY event_timestamp DESC;

-- How many events by status
SELECT status, COUNT(*) as count FROM detections GROUP BY status;

-- Search for a specific plate
SELECT * FROM detections WHERE plate_text = 'ABC1234';
```

---

## Run as a systemd service (recommended for Pi) [NOT YET TESTED AND VALIDATED]

```ini
# /etc/systemd/system/platewatcher.service
[Unit]
Description=LLM License Plate Extractor
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/platewatcher
Environment=PLATEWATCHER_APP_DIR=/home/pi/.local/state/platewatcher
Environment=PLATEWATCHER_EVENTS_DIR=/home/pi/.local/state/platewatcher/events
Environment=PLATEWATCHER_DB_URL=sqlite:////home/pi/.local/state/platewatcher/license_plates.db
Environment=PLATEWATCHER_LOG_FILE=/home/pi/.local/state/platewatcher/platewatcher.log
Environment=LLM_BASE_URL=http://localhost:11434
Environment=LLM_MODEL=moondream
ExecStart=/usr/bin/python3 /home/pi/platewatcher/main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable platewatcher
sudo systemctl start platewatcher
sudo systemctl status platewatcher
```
