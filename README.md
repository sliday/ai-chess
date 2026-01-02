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

## Recent Updates (v1.9.0)

- 🎉 **Game End Reactions**: Chat bots now react to wins/losses with personality-appropriate responses
- 🤖 **Chat Bot Model**: Switched to google/gemini-3-flash-preview for reliable responses
- 🎯 **Improved Modals**: "Start New Game" modal matches game over styling
- 🧹 **Data Cleanup**: Removed ~1,075 fraudulent short games from free models with inflated Elo

### v1.8.0

- ⏭️ **Skip Stuck Games**: Subtle "skip stuck game" link appears after 45s+ wait and 3+ status messages
- ⏱️ **Timeout Forfeit**: Slow models forfeit automatically, new game starts in 3 seconds
- 🤖 **Livelier Chat Bots**: Increased activity (40% on normal moves, 8-12s cooldowns)
- 🔧 **Bug Fixes**: Fixed disconnect cleanup, database connection, exception handling
- 🔗 **Standardized Headers**: All API calls use consistent HTTP-Referer

### v1.7.0
- 🔮 **Viewer Predictions**: Users can predict the winner before games end
- 💬 **Dual AI Chat Bots**: Two chat bots with distinct personalities (hype bot & skeptic)
- 💰 **Cost Tracking**: Real-time API cost tracking per player
- 🚫 **Model Blacklist**: Expensive models excluded from random selection

## Created By

[@stas_kulesh](https://x.com/stas_kulesh)

## License

Proprietary - All rights reserved
