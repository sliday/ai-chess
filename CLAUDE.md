# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Chess Arena (https://aichess.co) is a real-time chess platform where 349+ AI language models compete against each other via OpenRouter API. Features include live WebSocket updates, Elo-based leaderboard, move quality visualization, and game replay/GIF export.

**Tech Stack:**
- Backend: Python 3, FastAPI, WebSockets, python-chess
- Frontend: Vanilla JavaScript, p5.js, Tailwind CSS + DaisyUI
- Server: Gunicorn with Uvicorn workers, Nginx reverse proxy
- Database: SQLite
- AI API: OpenRouter (349+ models)
- Commentary: google/gemini-3-flash-preview (vision-capable)

## Directory Structure

```
/var/www/ai-chess/           # Production (USE THIS)
├── server.py                # FastAPI app, game logic, WebSocket, AI calls
├── database.py              # SQLite operations, Elo calculations, stats
├── index.html               # Frontend with p5.js board, WebSocket client
├── chess_games.db           # SQLite database
├── models.txt               # Available AI models list
├── static/pieces/           # SVG chess pieces
├── game.log                 # Application logs
└── gunicorn.log             # Server logs

/root/ai-chess/              # Legacy (reference only, outdated)
```

## Development Commands

### Server Management
```bash
# Restart server
pgrep -f "gunicorn.*server" | xargs sudo kill
cd /var/www/ai-chess
nohup gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000 > gunicorn.log 2>&1 &

# View logs
tail -f /var/www/ai-chess/game.log
tail -f /var/www/ai-chess/gunicorn.log
```

### Database
```bash
sqlite3 /var/www/ai-chess/chess_games.db

# Useful queries:
SELECT COUNT(*) FROM games;
SELECT * FROM model_stats ORDER BY elo DESC LIMIT 10;
SELECT * FROM games ORDER BY timestamp DESC LIMIT 5;
```

### Testing Endpoints
```bash
curl -s http://127.0.0.1:8000/stats | python3 -m json.tool
curl -s http://127.0.0.1:8000/stats/hourly | python3 -m json.tool
curl -s http://127.0.0.1:8000/leaderboard | python3 -m json.tool
```

### Setup (fresh install)
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key-here"
python3 -c "import database; database.init_db()"
```

## Architecture

### Backend (server.py)

**Core Classes:**
- `ChessGame`: Manages individual game state (board, moves, players)
- `GameManager`: Orchestrates games, WebSocket connections, autonomous gameplay

**Key Features:**
- Move validation with multi-tier retry system and hallucination detection
- Winner retention (winner stays for next game, swaps sides)
- Async commentary via vision-capable model
- Rate limiting with exponential backoff
- Model blacklist (expensive models like o3, gpt-5.2-pro excluded from random selection)
- Cost tracking per game for each player
- Dual AI chat bots with distinct personalities (hype bot & skeptic)
- Viewer predictions on game outcomes

**Move Validation Pipeline (server.py ~1150-1240):**
1. Hallucination detection (fast-fail if moving from empty square)
2. SAN to UCI conversion (Nf3 → g1f3)
3. Enhanced prompts with FEN and legal move count
4. Up to 3 model retries, then 3 API retries with fallback

### Frontend (index.html)

**Core Components:**
- p5.js chess board with piece animations
- WebSocket client for real-time updates
- Move quality visualization (brilliant !!, good !, mistake ?, blunder ??)
- Statistics dashboard with jQuery Sparklines
- Game replay and GIF generation

### Database Schema

**games table:**
- id, model1, model2, winner (0/1/NULL), result, reason
- moves (comma-separated UCI), move_count, timestamp
- elo_change_model1, elo_change_model2
- cost_white, cost_black (API costs per player)

**model_stats table:**
- model (PK), elo (default 1500), games, wins, losses, draws
- streak (e.g., 'WWW', 'LLD'), last_updated

**predictions table:**
- id, game_id, username, predicted_winner (0=white, 1=black)
- timestamp, correct (NULL until game ends)

## API Endpoints

- `POST /games` - Create new game (body: model1?, model2?, use_previous_result?)
- `GET /active_game` - Get current autonomous game
- `GET /games/{game_id}` - Get game state
- `GET /stats` - Platform statistics (period=daily|weekly, includes avg_cost_per_game, total_cost)
- `GET /stats/hourly` - 24-hour hourly data for sparklines
- `GET /leaderboard` - Model rankings (min_games=N)
- `GET /models` - Available AI models
- `POST /refresh-models` - Refresh model list from OpenRouter
- `POST /predict` - Submit viewer prediction (via WebSocket)
- `GET /predictions/{game_id}` - Get prediction counts for a game

## WebSocket Protocol

Connection: `ws://host/ws/{game_id}`

Message types: `game_state`, `move_made`, `commentary_update`, `status_update`, `game_over`

## Key Implementation Details

**Winner Retention:**
- Backend stores `last_game_result` for all games
- Frontend tracks `previousWinner` with model and position
- Winner marked with star, swaps sides (color) for rematch

**Move Quality (6 levels):**
- Brilliant!! (gold gradient), Good! (green), Interesting!? (blue)
- Dubious?! (orange), Mistake? (light red), Blunder?? (red gradient)
- Parsed from commentary, applied via CSS classes

**Statistics Caching:**
- 30-second TTL with daily/weekly cache keys
- Invalidation on game completion
- 99% reduction in database queries

**Reasoning Model Timeouts:**
- Extended timeout (180s) for models containing: o1, o3, r1, qwq, thinking, reasoning, step-1/2/3
- Regular timeout: 45s

**Model Blacklist:**
- Expensive models excluded from random selection: o3, o3-pro, gpt-5.2-pro, deep-research models
- Defined in `MODEL_BLACKLIST` set in server.py

**Dual Chat Bots:**
- Two AI bots with distinct personalities generated on server start
- Hype Bot: enthusiastic, uses caps for excitement, roots for underdogs
- Skeptic Bot: dry humor, questions everything, lowkey pessimistic
- Event-based probabilities: 70% on brilliant moves, 80% on blunders, 90% on game end
- Each bot has unique name (adjective + animal + emoji) and color

## Configuration

**Required Environment:**
- `OPENROUTER_API_KEY`: OpenRouter API key

**Optional:**
- `GAME_DELAY`: Delay between moves (default: 3s)
- `MAX_RETRIES`: Max move retries (default: 3)
- `MAX_TOKENS`: Max tokens per request (default: 50)
- `GAME_TIMEOUT`: Max game duration (default: 3600s)

## Known Issues

- Some free models have 8-10 move limits (auto-switch to reliable model)
- Model hallucinations (moving from empty squares) - fast-fail detection
- Rate limiting - retry logic with exponential backoff

## See Also

- `AICHESS.md` - Full technical documentation with line number references
- `README.md` - Project overview and features
