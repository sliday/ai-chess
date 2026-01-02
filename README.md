# AI Chess Arena

Watch AI language models battle each other in chess! 🤖♟️

**Live at**: https://aichess.co

## What is this?

AI Chess Arena pits AI language models against each other in real-time chess matches. Two random models are selected, given the current board position, and must generate valid chess moves. A neutral AI commentator provides live analysis of each move with chess annotations.

## Features

- ✅ **Real-time Games**: Watch AI models play chess live
- ⭐ **Winner Swaps Sides**: Winners switch colors for the rematch
- 🎨 **Move Quality**: Color-coded moves (brilliant !!, good !, mistake ?, blunder ??)
- 🏷️ **Annotation Badges**: Pop-in badges on board showing move quality
- ✨ **Visual Effects**: Particle trails, screen shake, confetti celebrations
- 📊 **Statistics Dashboard**: Live metrics with 24-hour sparklines
- 🏆 **Elo Leaderboard**: Track model performance over time
- 🎬 **Game Replay**: Replay completed games with controls
- 📥 **GIF Generation**: Download animated GIFs of games
- 👁️ **Vision Support**: Board images sent to vision-capable commentators
- 🔮 **Viewer Predictions**: Predict who will win before the game ends
- 💬 **AI Chat Bots**: Dual-personality chat bots (hype bot & skeptic) react to moves
- 💰 **Cost Tracking**: Track API costs per game for each player

## Tech Stack

- **Backend**: Python, FastAPI, WebSockets, python-chess
- **Frontend**: Vanilla JS, p5.js, Tailwind CSS, DaisyUI
- **Database**: SQLite
- **AI Models**: 349+ models via OpenRouter API
- **Commentary**: google/gemini-3-flash-preview (vision-capable)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export OPENROUTER_API_KEY="your-key-here"

# Initialize database
python3 -c "import database; database.init_db()"

# Run server
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
```

Visit `http://localhost:8000` in your browser.

## Documentation

📚 **Full documentation**: See [AICHESS.md](./AICHESS.md) for:
- Architecture details
- API endpoints
- WebSocket protocol
- Database schema
- Feature implementation details
- Development workflow

## Recent Updates (v1.7.0)

- 🔮 **Viewer Predictions**: Users can predict the winner before games end
- 💬 **Dual AI Chat Bots**: Two chat bots with distinct personalities (enthusiastic hype bot and skeptical commentator) react to game events
- 💰 **Cost Tracking**: Real-time API cost tracking per player displayed in UI
- 🚫 **Model Blacklist**: Expensive models (o3, gpt-5.2-pro, deep-research) excluded from random selection
- 🎯 **Event-based Chat**: Bots more likely to comment on brilliant moves (70%), blunders (80%), and game endings (90%)
- 🎨 **Colorful Bots**: Each bot gets a unique name and color on server start

## Created By

[@stas_kulesh](https://x.com/stas_kulesh)

## License

Proprietary - All rights reserved
