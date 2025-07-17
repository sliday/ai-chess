# AI Chess Arena

A web application where AI language models play chess against each other via the OpenRouter API. Watch powerful AI models compete in real-time with animated games, insightful commentary, and a performance leaderboard.

## Features

- AI vs AI chess matches using 300+ models from OpenRouter API
- Beautiful chess board visualization with Unicode chess pieces
- Real-time game updates with WebSockets
- Expert commentary from GPT-4.1-mini with chess annotation symbols (!, !!, ?, ??, etc.)
- Visual effects for brilliant moves, blunders, and checkmates
- Move history with SAN notation display
- Automatic interpretation for ambiguous or incorrectly formatted moves
- Leaderboard tracking model performance across games
- Game setup with model selection or random opponents
- Responsive interface using DaisyUI and TailwindCSS

## Screenshots

(Add screenshots here to showcase the UI)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/sliday/ai-chess.git
   cd ai-chess
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your OpenRouter API key as an environment variable:
   ```
   export OPENROUTER_API_KEY=your_api_key_here
   ```

4. Run the server:
   ```
   python server.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Technical Details

### Backend (FastAPI)

- **Chess Game Logic**: Built on Python's `chess` library for move validation and game state management
- **WebSocket API**: Provides real-time game updates to the client
- **OpenRouter Integration**: Sophisticated API integration with retry logic and fallback mechanisms
- **Move Interpretation**: Uses GPT-4.1-mini to interpret ambiguous moves into standard notation
- **Asynchronous Commentary**: Non-blocking game commentary that doesn't slow down gameplay
- **Leaderboard Management**: Tracks and ranks model performance based on game results

### Frontend (HTML/JS)

- **Responsive UI**: Built with DaisyUI and TailwindCSS
- **Interactive Chess Board**: Rendered with Unicode chess symbols
- **Visual Effects**: Animations for special moves and game events (confetti for wins, shaking for blunders)
- **WebSocket Client**: For receiving real-time game updates
- **Move History**: Formatted display of chess moves in standard algebraic notation

## API Endpoints

- `GET /` - Serves the main HTML interface
- `GET /models` - Returns the list of available AI models
- `POST /games` - Creates a new chess game between specified models
- `GET /games/{game_id}` - Returns the current state of a specific game
- `GET /leaderboard` - Returns the current model rankings
- `WebSocket /ws/{game_id}` - Real-time connection for game updates

## Configuration

- **models.txt**: Contains the list of OpenRouter model IDs available for games
- **results.txt**: Stores game results in CSV format for leaderboard persistence
- **requirements.txt**: Lists all Python dependencies

## Requirements

- Python 3.8+
- FastAPI and Uvicorn
- OpenRouter API key
- Modern web browser with WebSocket support

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Chess logic powered by [python-chess](https://python-chess.readthedocs.io/)
- UI components from [DaisyUI](https://daisyui.com/)
- Styling with [TailwindCSS](https://tailwindcss.com/)
- Models accessible via [OpenRouter](https://openrouter.ai/)