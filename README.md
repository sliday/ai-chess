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

## Tech Stack

- **Backend**: Python, FastAPI, WebSockets, python-chess
- **Frontend**: Vanilla JS, p5.js, Tailwind CSS, DaisyUI
- **Database**: SQLite
- **AI Models**: 353+ models via OpenRouter API
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

## URL Parameters

You can pre-select models via URL parameters:

```
https://aichess.co/?model=openai/gpt-4o&model2=anthropic/claude-sonnet-4
```

- `model` - White player (model1)
- `model2` - Black player (model2)
- Either can be omitted for random selection

## Documentation

📚 **Full documentation**: See [AICHESS.md](./AICHESS.md) for:
- Architecture details
- API endpoints
- WebSocket protocol
- Database schema
- Feature implementation details
- Development workflow

## Recent Updates (v1.6.0)

- 🏷️ Move annotation badges with pop-in animation on board
- ✨ Enhanced visual effects (particles, trails, screen shake)
- 🎉 Confetti celebration on game end
- 👁️ Vision model support for commentary (sends board images)
- 🔄 Winner now swaps sides (color) for rematch
- 🔴 Red particles for captures, tan for normal moves
- 🚗 Knight Rider scanner during AI thinking

## Created By

[@stas_kulesh](https://x.com/stas_kulesh)

## License

Proprietary - All rights reserved
