import os
import random
import json
import time
import chess
import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# Setup enhanced logging for the application (do not override uvicorn's logging)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Console log handler
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
logger.addHandler(_stream_handler)
# File log handler
_file_handler = logging.FileHandler("game.log")
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)
logger.info("AI Chess Arena starting up...")

# Initialize FastAPI app
app = FastAPI(title="AI Chess Arena")

# Read available models from models.txt and normalize IDs (remove leading slashes)
with open("models.txt", "r") as f:
    MODELS = [line.strip().lstrip('/') for line in f if line.strip()]

# Results file path
RESULTS_FILE = "results.txt"

# OpenRouter API key (get from environment variable)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set. API calls will fail.")

# Base URL for OpenRouter API
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Commentator model
COMMENTATOR_MODEL = "openai/gpt-4.1-mini"

# Chess annotation symbols
ANNOTATION_SYMBOLS = {
    "!!": "Brilliant move",
    "!": "Good move",
    "?!": "Questionable move",
    "?": "Mistake",
    "??": "Blunder",
    "!?": "Interesting move",
    "□": "With the idea",
    "⊕": "With an attack",
    "∞": "Unclear position",
    "=": "Equal position",
    "±": "White has a slight advantage",
    "∓": "Black has a slight advantage",
    "+−": "White has a decisive advantage",
    "−+": "Black has a decisive advantage",
    "+": "Check",
    "#": "Checkmate"
}

# Model for API requests
class ModelSelectionRequest(BaseModel):
    model1: Optional[str] = None
    model2: Optional[str] = None

# Class to represent a game result
class GameResult(BaseModel):
    model1: str
    model2: str
    winner: int  # 0 for model1, 1 for model2
    timestamp: str

# Class to manage the chess game with full state tracking
class ChessGame:
    def __init__(self, model1: str, model2: str):
        """Initialize a new chess game between two AI models
        
        Args:
            model1: The OpenRouter model ID for the white player
            model2: The OpenRouter model ID for the black player
        """
        # Initialize a standard chess board
        self.board = chess.Board()
        
        # Store AI model information
        self.model1 = model1  # White player
        self.model2 = model2  # Black player
        
        # Game state tracking
        self.current_player = 0  # 0 for model1 (white), 1 for model2 (black)
        self.moves_history = []  # List of moves in UCI format
        self.commentary = []     # Commentary from the GPT-4.1-mini observer
        self.status = "in_progress"  # Game status (in_progress or finished)
        self.winner = None       # 0 for model1, 1 for model2, None for draw
        self.reason = None       # Reason for game ending (checkmate, stalemate, etc.)
        
        logger.info(f"New game created: {model1} (White) vs {model2} (Black)")
    
    def get_current_model(self) -> str:
        return self.model1 if self.current_player == 0 else self.model2
    
    def get_opponent_model(self) -> str:
        return self.model2 if self.current_player == 0 else self.model1
    
    def get_board_ascii(self) -> str:
        return str(self.board)
    
    def get_fen(self) -> str:
        return self.board.fen()
    
    def apply_move(self, move_str: str) -> bool:
        try:
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.moves_history.append(move_str)
                self.current_player = 1 - self.current_player  # Switch player
                return True
            else:
                self.status = "finished"
                self.winner = 1 - self.current_player  # Current player loses
                self.reason = "illegal_move"
                return False
        except ValueError:
            self.status = "finished"
            self.winner = 1 - self.current_player  # Current player loses
            self.reason = "invalid_notation"
            return False
    
    def is_game_over(self) -> bool:
        if self.status == "finished":
            return True
        
        if self.board.is_game_over():
            self.status = "finished"
            
            if self.board.is_checkmate():
                # In checkmate, the player who just moved wins
                self.winner = 1 - self.current_player
                self.reason = "checkmate"
            elif self.board.is_stalemate():
                self.winner = None
                self.reason = "stalemate"
            elif self.board.is_insufficient_material():
                self.winner = None
                self.reason = "insufficient_material"
            elif len(self.moves_history) >= 70 or self.board.is_seventyfive_moves():
                # After 70 moves (35 per player), decide the winner based on material
                if len(self.moves_history) >= 70:
                    self.reason = "reached_move_limit"
                    # Calculate material for each side
                    material = self.calculate_material()
                    white_material = material['white']
                    black_material = material['black']
                    
                    # Determine winner based on material advantage
                    if white_material > black_material:
                        self.winner = 0  # White wins
                        logger.info(f"White wins on material after move limit: {white_material} vs {black_material}")
                    elif black_material > white_material:
                        self.winner = 1  # Black wins
                        logger.info(f"Black wins on material after move limit: {black_material} vs {white_material}")
                    else:
                        self.winner = None  # Draw if equal material
                        logger.info(f"Draw on material after move limit: {white_material} vs {black_material}")
                else:
                    self.winner = None
                    self.reason = "seventyfive_moves"
            elif self.board.is_fivefold_repetition():
                self.winner = None
                self.reason = "fivefold_repetition"
            else:
                self.winner = None
                self.reason = "draw"
                
            return True
        
        return False
    
    def get_result(self) -> Dict[str, Any]:
        if not self.is_game_over():
            return {"status": "in_progress"}
        
        if self.winner is None:
            return {
                "status": "finished",
                "result": "draw",
                "reason": self.reason
            }
        
        winner_model = self.model1 if self.winner == 0 else self.model2
        return {
            "status": "finished",
            "result": "win",
            "winner": self.winner,
            "winner_model": winner_model,
            "reason": self.reason
        }
    
    def calculate_material(self) -> Dict[str, int]:
        """Calculate material value for each player
        
        Standard piece values:
        - Pawn: 1
        - Knight/Bishop: 3
        - Rook: 5
        - Queen: 9
        - King: not counted
        
        Returns:
            Dictionary with 'white' and 'black' material values
        """
        piece_values = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0
        }
        
        white_material = 0
        black_material = 0
        
        board_str = str(self.board)
        for char in board_str:
            if char in piece_values:
                if char.isupper():  # White piece
                    white_material += piece_values[char]
                else:  # Black piece
                    black_material += piece_values[char]
        
        return {'white': white_material, 'black': black_material}
        
    def board_to_unicode(self) -> List[List[dict]]:
        """Convert chess board to unicode representation with filled pieces and color information
        
        Returns:
            A 2D array of dictionaries, each containing:
            - symbol: The Unicode symbol for the piece
            - color: 'w' for white, 'b' for black, or empty for empty squares
        """
        # Use filled Unicode chess symbols for all pieces
        unicode_pieces = {
            # Black pieces (filled symbols)
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            # White pieces (using the same filled symbols)
            'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',
            '.': ''  # Use empty string instead of space for empty squares
        }
        
        # Map piece letters to their colors
        color_map = {
            # Lowercase letters are black pieces
            'r': 'b', 'n': 'b', 'b': 'b', 'q': 'b', 'k': 'b', 'p': 'b',
            # Uppercase letters are white pieces
            'R': 'w', 'N': 'w', 'B': 'w', 'Q': 'w', 'K': 'w', 'P': 'w',
        }
        
        # Convert board to object format with proper color information
        board_data = []
        
        # Get FEN representation for more reliable piece identification
        fen = self.board.fen()
        fen_pieces = fen.split(' ')[0]  # Get just the piece placement part
        fen_rows = fen_pieces.split('/')
        
        # Process each row from FEN
        for row_idx, fen_row in enumerate(fen_rows):
            board_row = []
            col_idx = 0
            
            for char in fen_row:
                if char.isdigit():
                    # Empty squares
                    empty_count = int(char)
                    for _ in range(empty_count):
                        board_row.append({
                            "symbol": "",
                            "color": ""
                        })
                        col_idx += 1
                else:
                    # Piece - add with proper color information
                    color = 'w' if char.isupper() else 'b'
                    board_row.append({
                        "symbol": unicode_pieces.get(char, ''),
                        "color": color
                    })
                    col_idx += 1
            
            board_data.append(board_row)
        
        return board_data

# Game manager to handle active games and the leaderboard
class GameManager:
    """
    Central manager for all games and game results in the AI Chess Arena.
    
    This class is responsible for:
    - Managing all active games
    - Loading and saving game results
    - Creating and ending games
    - Computing the leaderboard statistics
    - Normalizing model IDs for consistent handling
    """
    def __init__(self):
        """Initialize the game manager and load previous game results"""
        # Dictionary of active games, keyed by game_id
        self.active_games: Dict[str, ChessGame] = {}
        
        # Load previous game results from file
        self.game_results: List[GameResult] = self.load_results()
        logger.info(f"Loaded {len(self.game_results)} previous game results from {RESULTS_FILE}")
    
    def load_results(self) -> List[GameResult]:
        """Load game results from file"""
        results = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(",")
                        if len(parts) == 4:
                            results.append(GameResult(
                                model1=parts[0],
                                model2=parts[1],
                                winner=int(parts[2]),
                                timestamp=parts[3]
                            ))
        return results
    
    def save_result(self, result: GameResult):
        """Save game result to file"""
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{result.model1},{result.model2},{result.winner},{result.timestamp}\n")
        self.game_results.append(result)
    
    def create_game(self, model1: Optional[str] = None, model2: Optional[str] = None) -> str:
        """Create a new game with specified or random models"""
        if not model1 or model1 == "random":
            model1 = random.choice(MODELS)
        
        if not model2 or model2 == "random":
            # Make sure we don't have model playing against itself
            available_models = [m for m in MODELS if m != model1]
            model2 = random.choice(available_models)
        
        # Ensure consistent model ID formatting (no leading slashes)
        normalized_model1 = model1.lstrip('/') if model1 else None
        normalized_model2 = model2.lstrip('/') if model2 else None
        
        game_id = f"game_{int(time.time())}_{random.randint(1000, 9999)}"
        self.active_games[game_id] = ChessGame(normalized_model1, normalized_model2)
        return game_id
    
    def get_game(self, game_id: str) -> ChessGame:
        """Get a game by ID"""
        if game_id not in self.active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        return self.active_games[game_id]
    
    def end_game(self, game_id: str):
        """End a game and save results if needed"""
        if game_id in self.active_games:
            game = self.active_games[game_id]
            if game.status == "finished" and game.winner is not None:
                # Save result
                result = GameResult(
                    model1=game.model1,
                    model2=game.model2,
                    winner=game.winner,
                    timestamp=datetime.now().isoformat()
                )
                self.save_result(result)
            
            # Remove game
            del self.active_games[game_id]
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Generate a leaderboard based on game results"""
        model_stats = {}
        
        # Initialize stats for all models with normalized IDs (no leading slashes)
        for model in MODELS:
            normalized_model = model.lstrip('/')
            model_stats[normalized_model] = {
                "model": normalized_model,
                "display_name": self._get_display_name(normalized_model),
                "games": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0
            }
        
        # Compile stats from results
        for result in self.game_results:
            # Normalize model IDs by removing leading slashes
            model1 = result.model1.lstrip('/')
            model2 = result.model2.lstrip('/')
            
            # If the model isn't in our stats yet (new model), add it
            if model1 not in model_stats:
                model_stats[model1] = {
                    "model": model1,
                    "display_name": self._get_display_name(model1),
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0
                }
                
            if model2 not in model_stats:
                model_stats[model2] = {
                    "model": model2,
                    "display_name": self._get_display_name(model2),
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0
                }
            
            # Update model1 stats
            model_stats[model1]["games"] += 1
            if result.winner == 0:
                model_stats[model1]["wins"] += 1
            else:
                model_stats[model1]["losses"] += 1
            
            # Update model2 stats
            model_stats[model2]["games"] += 1
            if result.winner == 1:
                model_stats[model2]["wins"] += 1
            else:
                model_stats[model2]["losses"] += 1
        
        # Calculate win rates
        for model, stats in model_stats.items():
            if stats["games"] > 0:
                stats["win_rate"] = round(stats["wins"] / stats["games"] * 100, 2)
        
        # Convert to list and sort by win rate (for models with at least 1 game)
        leaderboard = [stats for model, stats in model_stats.items() if stats["games"] > 0]
        leaderboard.sort(key=lambda x: (x["win_rate"], x["wins"]), reverse=True)
        
        return leaderboard
    
    def _get_display_name(self, model_id: str) -> str:
        """Generate a friendly display name for a model ID"""
        if not model_id:
            return "Unknown Model"
        
        # Remove leading slash if present
        clean_id = model_id.lstrip('/')
        
        # Split into provider and model parts
        parts = clean_id.split('/')
        if len(parts) < 2:
            return clean_id
        
        provider = parts[0]
        model_name = parts[1]
        
        # Remove version tag for display
        if ':' in model_name:
            model_name = model_name.split(':')[0]
            
        return f"{provider} / {model_name}"

# Initialize game manager
game_manager = GameManager()

# Function to call OpenRouter API
async def interpret_move(raw_move: str, board_state: str, current_player: int = 0) -> str:
    """Use direct parsing and GPT-4.1-mini to interpret ambiguous or incorrectly formatted moves into UCI format
    
    Args:
        raw_move: The raw move string from the model
        board_state: The current board state (ASCII or FEN)
        current_player: Which player is making the move (0 for white, 1 for black)
    
    Returns:
        UCI move string or "INVALID"
    """
    if not raw_move:
        return "INVALID"
    
    # Clean up the raw move (remove extra spaces, punctuation, newlines)
    raw_move = raw_move.strip().replace('.', '').replace(',', '').replace(':', '')
    
    # Truncate if the response contains newlines or other text
    if '\n' in raw_move:
        # Take just the first line which should contain the move
        first_line = raw_move.split('\n')[0].strip()
        logger.info(f"Model returned multiline response. Using first line: '{first_line}'")
        raw_move = first_line
    
    # Remove piece prefixes (like 'p' in 'pe2e4') to ensure standard UCI format
    if len(raw_move) == 5 and raw_move[0].lower() in 'prnbqk' and raw_move[1] in 'abcdefgh' and raw_move[2] in '12345678' and raw_move[3] in 'abcdefgh' and raw_move[4] in '12345678':
        # This looks like a prefixed UCI move (pe2e4) - remove the piece prefix
        raw_move = raw_move[1:]
        logger.info(f"Removed piece prefix from move: {raw_move}")
    
    # Which side is making the move? (used for both pawn move expansion and FEN creation)
    active_color = 'b' if current_player == 1 else 'w'
    
    # Try to parse a chess board from the board state
    try:
        # Start with a default board with the correct active player
        current_board = chess.Board()
        if active_color == 'b':
            # Set black as active player with a null move
            current_board.push(chess.Move.null())
        
        # ASCII board detection - if it contains newlines and typical ASCII board characters
        ascii_board_pattern = '\n' in board_state and ('r ' in board_state or 'p ' in board_state or 'R ' in board_state)
        
        # If it looks like an ASCII board, just use our default board with correct color
        if ascii_board_pattern:
            logger.info(f"Using default board with active color: {active_color}")
            # Already initialized with correct color above
        else:
            # It might be a FEN string - try to parse it
            try:
                # Clean up any extra whitespace
                clean_fen = board_state.strip()
                
                # Make sure it has a color indicator and all FEN components
                if ' w ' not in clean_fen and ' b ' not in clean_fen:
                    # This appears to be just the piece placement part of FEN
                    # Add the other required FEN components including active color
                    clean_fen = f"{clean_fen} {active_color} KQkq - 0 1"
                
                test_board = chess.Board(clean_fen)
                # If we got here, the FEN was valid
                current_board = test_board
                logger.info(f"Successfully created board from FEN with active color: {active_color}")
            except Exception as e:
                logger.warning(f"Could not parse as FEN: {e} - using default board")
                # Keep the default board we already created
    except Exception as board_err:
        logger.error(f"Error creating chess board: {board_err}")
        # Fall back to default board with correct turn
        current_board = chess.Board()
        if active_color == 'b':
            current_board.push(chess.Move.null())
    
    # Step 1: Check if it's already a valid UCI move
    if len(raw_move) == 4 and raw_move[0] in "abcdefgh" and raw_move[1] in "12345678" and \
       raw_move[2] in "abcdefgh" and raw_move[3] in "12345678":
        try:
            move = chess.Move.from_uci(raw_move)
            if move in current_board.legal_moves:
                return raw_move
        except ValueError:
            pass  # Continue to other checks if this fails
    
    # Step 2: Handle common pawn move shorthand (e.g., "e4" means "e2e4")
    if len(raw_move) == 2 and raw_move[0] in "abcdefgh" and raw_move[1] in "12345678":
        for row in [2, 7]:  # White pawns start at row 2, black at row 7
            possible_uci = f"{raw_move[0]}{row}{raw_move}"
            try:
                move = chess.Move.from_uci(possible_uci)
                if move in current_board.legal_moves:
                    logger.info(f"Expanded pawn shorthand {raw_move} to {possible_uci}")
                    return possible_uci
            except ValueError:
                continue
    
    # Step 3: Try to parse as SAN and convert to UCI
    try:
        # The correct method is board.parse_san(), not chess.Move.from_san()
        move = current_board.parse_san(raw_move)
        uci_move = move.uci()
        logger.info(f"Converted SAN {raw_move} to UCI {uci_move}")
        return uci_move
    except ValueError:
        pass  # Continue to API interpretation if direct parsing fails
    
    # If all direct parsing attempts fail, use GPT-4.1-mini to interpret
    logger.info(f"Attempting to interpret ambiguous move with API: '{raw_move}'")
    
    # Get proper FEN from board
    current_fen = current_board.fen()
    
    # Normalize model ID: remove leading slash if present
    interpreter_model = "openai/gpt-4.1-mini"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://ai-chess-arena.example.com",
        "Content-Type": "application/json"
    }
    
    # Zero-shot payload with direct instructions for move interpretation
    data = {
        "model": interpreter_model,
        "messages": [
            {
                "role": "system", 
                "content": "You are a chess move interpreter. Output only a 4-character UCI move or 'INVALID'."
            },
            {
                "role": "user", 
                "content": f"""
Board FEN: {current_fen}
Active player: {"WHITE" if active_color == 'w' else "BLACK"}
Move to interpret: "{raw_move}"

Respond with ONLY the 4-character UCI format (e.g., "e2e4") or exactly "INVALID".
No explanation, no additional text.
"""
            }
        ],
        "max_tokens": 10
    }
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            logger.info(f"Sending interpretation request to {interpreter_model} for move: '{raw_move}'")
            response = await client.post(OPENROUTER_API_URL, json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                interpretation = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Interpreted move '{raw_move}' as '{interpretation}'")
                
                # If the interpretation is "INVALID" or doesn't look like a UCI move, return invalid
                if interpretation == "INVALID" or not (
                    len(interpretation) == 4 and 
                    interpretation[0] in "abcdefgh" and 
                    interpretation[1] in "12345678" and
                    interpretation[2] in "abcdefgh" and 
                    interpretation[3] in "12345678"
                ):
                    return "INVALID"
                
                # Validate the UCI move with the current board
                try:
                    move = chess.Move.from_uci(interpretation)
                    if move in current_board.legal_moves:
                        return interpretation
                    else:
                        logger.warning(f"Interpreted move '{interpretation}' is not legal on the current board")
                        return "INVALID"
                except Exception as validation_error:
                    logger.warning(f"Interpreted move '{interpretation}' is not valid UCI: {validation_error}")
                    return "INVALID"
            else:
                logger.warning(f"No interpretation received for '{raw_move}'")
                return "INVALID"
                
        except httpx.TimeoutException:
            logger.error(f"Timeout interpreting move '{raw_move}'")
            return "INVALID"
            
        except Exception as e:
            logger.error(f"Error interpreting move '{raw_move}': {e}")
            # Log the response body if it's an HTTP error
            if isinstance(e, httpx.HTTPStatusError) and e.response:
                logger.error(f"Interpreter Error Response: {e.response.text}")
            return "INVALID"

async def call_model(model: str, prompt: str, max_retries: int = 3) -> str:
    """Call a model via OpenRouter API with improved retry logic and model ID handling
    
    This function handles all aspects of communicating with the OpenRouter API including:
    - Normalizing model IDs for consistent handling
    - Special handling for model suffixes (like ":extended")
    - Structured prompting with system and user messages
    - Sophisticated retry logic with progressive fallbacks
    - Robust error handling and logging
    
    Args:
        model: The OpenRouter model ID to use
        prompt: The chess-related prompt to send to the model
        max_retries: Maximum number of retry attempts before giving up
        
    Returns:
        The model's response as a string, or empty string if all attempts fail
    """
    # Normalize model ID: remove leading slash if present
    model_id = model.lstrip('/')
    logger.info(f"Calling OpenRouter API with model: {model_id}")
    
    # Count how many moves have been made by checking the prompt
    moves_count = 0
    if "Previous moves:" in prompt:
        moves_section = prompt.split("Previous moves:")[1].split("\n\n")[0]
        moves_count = moves_section.count(".")
        # For BLACK's turn, reduce by 0.5 since we're counting complete turns
        if "BLACK" in prompt:
            moves_count = moves_count - 0.5
        logger.info(f"Detected {moves_count} move pairs already made in the game")
    
    # Some models have move limits (especially free versions)
    # If we're approaching move limits, use a more reliable model immediately
    if moves_count >= 8 and (":free" in model_id or model_id.endswith("-1.5")):
        logger.warning(f"Approaching potential move limit ({moves_count} moves made) with model {model_id}. Switching to a more reliable model.")
        model_id = "openai/gpt-4.1-mini"
    
    # Standard headers required by OpenRouter
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://ai-chess-arena.example.com",
        "Content-Type": "application/json"
    }
    
    # Handle models with special suffixes (like ":extended", ":thinking", ":free")
    base_model_id = model_id
    model_suffix = ""
    if ":" in model_id:
        base_model_id, model_suffix = model_id.split(":", 1)
        model_suffix = ":" + model_suffix
        logger.info(f"Model has suffix: base={base_model_id}, suffix={model_suffix}")
    
    # Enhanced payload with zero-shot system message for direct responses
    system_message = "You are a chess engine. Respond with only a valid chess move in UCI notation (e.g. 'e2e4', 'g1f3'). No explanations, no other text."
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    }
    
    # Progressive fallback prompts with direct zero-shot instructions
    fallback_prompts = [
        "No explanation needed. Respond with only a valid chess move in UCI format (e.g., 'e2e4'). Your move:",
        "Provide just the 4-character UCI chess move notation. Example format: 'e2e4', 'g1f3'. Your move:",
        "Reply with only four characters representing your move in UCI format. Your move:"
    ]
    
    # Track if we've tried the base model without suffix for models like "model:extended"
    tried_base_model = False
    
    # Implement enhanced retry logic with multiple fallback strategies
    for attempt in range(max_retries + 1):
        current_model = model_id
        
        # For retries, modify the approach
        if attempt > 0:
            logger.info(f"Retry attempt {attempt}/{max_retries} for model {current_model}")
            
            # Simplify the prompt on progressive retries
            if attempt <= len(fallback_prompts):
                fallback_index = attempt - 1
                data["messages"] = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": fallback_prompts[fallback_index]}
                ]
                logger.info(f"Using fallback prompt {fallback_index+1}: {fallback_prompts[fallback_index][:30]}...")
            
            # On later retries, try the base model without suffix if we haven't already
            if attempt == max_retries - 1 and model_suffix and not tried_base_model:
                current_model = base_model_id
                data["model"] = current_model
                tried_base_model = True
                logger.info(f"Trying base model without suffix: {current_model}")
            
            # On final retry, fall back to a reliable model with direct zero-shot prompt
            if attempt == max_retries:
                current_model = "openai/gpt-4.1-mini"
                data["model"] = current_model
                data["messages"] = [
                    {"role": "system", "content": "You are a chess engine. Output must be ONLY a valid 4-character UCI chess move."},
                    {"role": "user", "content": f"""
You are playing as {"WHITE" if 'WHITE' in prompt else "BLACK"}.
Current board:
{prompt}

Output only the exact 4-character UCI notation (e.g., 'e2e4', 'g1f3'). No explanation or additional text.
Your move:
"""}
                ]
                logger.info(f"Final fallback to reliable model: {current_model}")

        # Log the request payload
        logger.info(f"OpenRouter Request for {current_model} (attempt {attempt+1}): {json.dumps(data)}")

        try:
            # Use a longer timeout for the HTTP client
            async with httpx.AsyncClient(timeout=httpx.Timeout(45.0)) as client:
                # Send the request as JSON (not as a string)
                response = await client.post(OPENROUTER_API_URL, json=data, headers=headers)
                response.raise_for_status()
                response_data = response.json()

                # Check for empty content or missing message content - common with move limits
                no_content = False
                if ("choices" in response_data and len(response_data["choices"]) > 0 and 
                    (not response_data["choices"][0].get("message", {}).get("content") or 
                     response_data["choices"][0].get("message", {}).get("content", "").strip() == "")):
                    no_content = True
                    logger.warning(f"Empty response from {current_model}. This may indicate a move limit has been reached.")
                    
                    # Immediately move to a more reliable model on empty response
                    if current_model != "openai/gpt-4.1-mini" and (attempt < max_retries):
                        current_model = "openai/gpt-4.1-mini"
                        data["model"] = current_model
                        data["messages"] = [
                            {"role": "system", "content": "You are a chess engine. Output must be ONLY a valid 4-character UCI chess move."},
                            {"role": "user", "content": prompt}
                        ]
                        logger.info(f"Switching to reliable model due to empty response: {current_model}")
                        # Skip to next attempt without counting this as a retry
                        continue
                    elif attempt >= max_retries:
                        return ""

                # Log partial response to avoid excessive logging
                log_response = {
                    "id": response_data.get("id", "unknown"),
                    "model": response_data.get("model", "unknown"),
                    "choices": [{"index": c.get("index"), "message": {"role": c.get("message", {}).get("role")}} 
                                for c in response_data.get("choices", [])]
                }
                logger.info(f"OpenRouter response for {current_model} (attempt {attempt+1}): {json.dumps(log_response)}")

                if "choices" in response_data and len(response_data["choices"]) > 0 and not no_content:
                    move = response_data["choices"][0]["message"]["content"].strip()
                    # Allow slightly longer response to handle more varied formats
                    return move[:30]
                else:
                    logger.warning(f"No valid response content from {current_model} (attempt {attempt+1})")
                    # Try again if we have retries left
                    if attempt < max_retries:
                        await asyncio.sleep(1.5)  # Slightly longer pause before retry
                        continue
                    return ""

        except httpx.TimeoutException:
            logger.error(f"Timeout calling model {current_model} (attempt {attempt+1})")
            if attempt < max_retries:
                await asyncio.sleep(1.5)  # Slightly longer pause before retry
                continue
            return ""

        except Exception as e:
            logger.error(f"Error calling model {current_model} (attempt {attempt+1}): {e}")
            # Log the response body if it's an HTTP error
            if isinstance(e, httpx.HTTPStatusError) and e.response:
                logger.error(f"OpenRouter Error Response: {e.response.text}")
            
            # If there's an HTTP 400 error related to the model parameter, log it specially
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", "")
                    if "model" in error_msg.lower():
                        logger.error(f"Model-related error: {error_msg}")
                except:
                    pass
            
            if attempt < max_retries:
                await asyncio.sleep(1.5)  # Slightly longer pause before retry
                continue
            return ""
    
    # If we've exhausted all retries
    logger.error(f"All retry attempts failed for model {model_id}")
    return ""

async def get_commentary(board_state: str, last_move: str, moves_history: list = None) -> str:
    """Get chess commentary from GPT-4o-mini asynchronously without blocking the game"""
    # Create a task for the commentary but don't await it immediately
    return asyncio.create_task(_fetch_commentary(board_state, last_move, moves_history))

async def generate_friendly_game_over_reason(reason: str, winner_model: str, result_type: str) -> str:
    """Generate a user-friendly explanation for why the game ended
    
    Args:
        reason: Technical reason the game ended (e.g., 'invalid_notation', 'checkmate')
        winner_model: The model that won (if any)
        result_type: Type of result ('win', 'draw', etc.)
        
    Returns:
        A user-friendly explanation of the game outcome
    """
    # Normalize model ID: remove leading slash if present
    commentator_model = COMMENTATOR_MODEL.lstrip('/')
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://ai-chess-arena.example.com",
        "Content-Type": "application/json"
    }
    
    # Map technical reasons to types of explanations needed
    explanation_prompt = ""
    if reason == "invalid_notation":
        explanation_prompt = "Explain why the model failed to make a valid chess move (friendly explanation for non-experts)."
    elif reason == "illegal_move":
        explanation_prompt = "Explain why a model might attempt an illegal chess move (friendly explanation for non-experts)."
    elif reason == "checkmate":
        explanation_prompt = "Describe the victory by checkmate in fun, simple terms."
    elif reason == "reached_move_limit":
        explanation_prompt = "Explain why the game ended after reaching the 70-move limit and how a winner was determined based on material."
    elif reason in ["stalemate", "insufficient_material", "seventyfive_moves", "fivefold_repetition"]:
        explanation_prompt = f"Explain what '{reason}' means in chess in simple, friendly terms."
    else:
        explanation_prompt = "Provide a simple, friendly explanation for how this chess game ended."
    
    # Enhanced payload with system and user messages
    data = {
        "model": commentator_model,
        "messages": [
            {
                "role": "system",
                "content": """You write engaging, simple explanations for chess game outcomes. Keep your response to 1-2 sentences, friendly and accessible to chess beginners."""
            },
            {
                "role": "user",
                "content": f"""
The chess game just ended with the following outcome:
- Result: {result_type}
- Winner: {winner_model if result_type == 'win' else 'None (Draw)'}
- Technical reason: {reason}

{explanation_prompt}
"""
            }
        ],
        "max_tokens": 100
    }
    
    logger.info(f"Requesting game over explanation from {commentator_model}")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            response = await client.post(OPENROUTER_API_URL, json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                explanation = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Received game over explanation: {explanation[:50]}...")
                return explanation
            else:
                logger.warning(f"No valid game over explanation received")
                return f"Game ended due to {reason}."
        except Exception as e:
            logger.error(f"Error getting game over explanation: {e}")
            return f"Game ended due to {reason}."

async def _fetch_commentary(board_state: str, last_move: str, moves_history: list = None) -> str:
    """Internal function to fetch commentary from the model using improved API format"""
    # Normalize model ID: remove leading slash if present
    commentator_model = COMMENTATOR_MODEL.lstrip('/')
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://ai-chess-arena.example.com",
        "Content-Type": "application/json"
    }
    
    # Include move history if available
    moves_context = ""
    if moves_history and len(moves_history) > 0:
        # Format move history in pairs
        formatted_moves = []
        for i in range(0, len(moves_history), 2):
            move_num = i // 2 + 1
            white_move = moves_history[i]
            black_move = moves_history[i+1] if i+1 < len(moves_history) else ""
            if black_move:
                formatted_moves.append(f"{move_num}. {white_move} {black_move}")
            else:
                formatted_moves.append(f"{move_num}. {white_move}")
        
        moves_context = f"\nMove history: {' '.join(formatted_moves)}"
    
    # Enhanced payload with system and user messages
    data = {
        "model": commentator_model,
        "messages": [
            {
                "role": "system",
                "content": """You are a concise chess commentator. Your analysis should be 1-2 sentences, use standard chess notation, and include appropriate annotation symbols. Be insightful but brief."""
            },
            {
                "role": "user",
                "content": f"""
Provide a brief analysis of the latest chess move and current position.

Current board state:
{board_state}{moves_context}

Last move: {last_move}

Use these chess annotation symbols when appropriate:
!! (brilliant move), ! (good move), ?! (questionable move), ? (mistake), ?? (blunder), !? (interesting move),
⊕ (with an attack), ∞ (unclear position), = (equal position), 
± (White has a slight advantage), ∓ (Black has a slight advantage),
+− (White has a decisive advantage), −+ (Black has a decisive advantage)
"""
            }
        ],
        "max_tokens": 100
    }
    
    logger.info(f"Requesting commentary from {commentator_model}")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            response = await client.post(OPENROUTER_API_URL, json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                commentary = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Received commentary from {commentator_model}: {commentary[:30]}...")
                return commentary
            else:
                logger.warning(f"No valid commentary received from {commentator_model}")
                return "No commentary available."
        except httpx.TimeoutException:
            logger.error(f"Timeout getting commentary from {commentator_model}")
            return "Commentary unavailable due to timeout."
        except Exception as e:
            logger.error(f"Error getting commentary from {commentator_model}: {e}")
            # Log the response body if it's an HTTP error
            if isinstance(e, httpx.HTTPStatusError) and e.response:
                logger.error(f"Commentary Error Response: {e.response.text}")
            return "Commentary unavailable due to an error."

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the index.html file"""
    with open("index.html", "r") as f:
        return f.read()

@app.get("/models")
async def get_models():
    """Get list of available models"""
    # Models are already normalized (no leading slashes)
    # For display purposes, include a formatted name
    formatted_models = []
    for model_id in MODELS:
        # Extract provider and model name
        parts = model_id.split('/')
        if len(parts) >= 2:
            provider = parts[0]
            # Remove version tag for display
            name = parts[1].split(':')[0] if ':' in parts[1] else parts[1]
            display_name = f"{provider} / {name}"
        else:
            display_name = model_id
            
        formatted_models.append({
            "id": model_id,
            "display_name": display_name
        })
        
    return {"models": formatted_models}

@app.post("/games")
async def create_game(request: ModelSelectionRequest):
    """Create a new game"""
    game_id = game_manager.create_game(request.model1, request.model2)
    return {"game_id": game_id}

@app.get("/games/{game_id}")
async def get_game(game_id: str):
    """Get game state"""
    game = game_manager.get_game(game_id)
    return {
        "game_id": game_id,
        "model1": game.model1,
        "model2": game.model2,
        "current_player": game.current_player,
        "board": game.get_board_ascii(),
        "moves": game.moves_history,
        "status": game.status,
        "result": game.get_result(),
        "commentary": game.commentary
    }

@app.get("/leaderboard")
async def get_leaderboard():
    """Get model leaderboard"""
    return {"leaderboard": game_manager.get_leaderboard()}

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket connection for real-time game updates and automated game play
    
    This is the core of the AI Chess Arena - it handles:
    1. Establishing and maintaining the WebSocket connection
    2. Running the main game loop where models make moves
    3. Sending real-time updates to the client
    4. Handling game completion and cleanup
    5. Managing asynchronous commentary from GPT-4.1-mini
    
    Args:
        websocket: The WebSocket connection to the client
        game_id: The ID of the game to play
    """
    # Accept the WebSocket connection
    await websocket.accept()
    logger.info(f"WebSocket connection established for game {game_id}")
    
    try:
        # Get the game from the game manager
        game = game_manager.get_game(game_id)
        logger.info(f"Starting game: {game.model1} vs {game.model2}")
        
        # Function to safely send JSON through websocket, handling client disconnect
        async def safe_send_json(data):
            try:
                await websocket.send_json(data)
                return True
            except WebSocketDisconnect:
                logger.info(f"Client disconnected during message send for game {game_id}")
                return False
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                return False
        
        # Send initial game state to the client
        if not await safe_send_json({
            "type": "game_state",
            "data": {
                "game_id": game_id,
                "model1": game.model1,
                "model2": game.model2,
                "current_player": game.current_player,
                "board": game.board_to_unicode(),
                "moves": game.moves_history,
                "status": game.status,
                "result": game.get_result(),
                "commentary": game.commentary
            }
        }):
            # Client disconnected early, clean up and return
            game_manager.end_game(game_id)
            return
        
        # Main game loop - runs until the game is over
        while not game.is_game_over():
            # Get the current player's model
            current_model = game.get_current_model()
            
            # Prepare the chess board state as a text representation for the model
            board_state = game.get_board_ascii()
            
            # Format previous moves history for context
            moves_history_text = ""
            if game.moves_history:
                # Convert UCI moves to a readable format with numbering
                moves_formatted = []
                for i, move in enumerate(game.moves_history):
                    move_num = (i // 2) + 1
                    if i % 2 == 0:  # White's move
                        moves_formatted.append(f"{move_num}. {move}")
                    else:  # Black's move
                        # Append to previous white's move
                        moves_formatted[-1] += f" {move}"
                
                moves_history_text = "Previous moves:\n" + "\n".join(moves_formatted)
            
            # Build the zero-shot prompt for the AI model with direct instruction
            prompt = f"""
You are playing as {"WHITE" if game.current_player == 0 else "BLACK"} in this chess game.
{"It is YOUR TURN to move now." if moves_history_text else "You are making the first move of the game."}

Current board position (YOUR pieces are {"UPPERCASE" if game.current_player == 0 else "lowercase"}):
{board_state}

{moves_history_text}

No intro, no outro, only your next move in proper UCI format (e.g., "e2e4" for moving from e2 to e4). Your next move is:
"""
            
            # Send status update to the client
            if not await safe_send_json({
                "type": "status_update",
                "data": {
                    "message": f"Waiting for {current_model} to make a move...",
                    # Include latest commentary if available
                    "commentary": game.commentary[-1] if game.commentary else ""
                }
            }):
                # Client disconnected, clean up and return
                game_manager.end_game(game_id)
                return
            
            # Call the AI model through OpenRouter API to get its chess move
            # call_model handles all API communication, retries, and fallbacks
            raw_move = await call_model(current_model, prompt)
            
            # Log the raw response for debugging purposes
            logger.info(f"Received raw move '{raw_move}' from model {current_model}")
            
            # Handle the case where we couldn't get any response from the model
            if not raw_move:
                logger.error(f"Failed to get any move from model {current_model} after all retries")
                game.status = "finished"
                game.winner = 1 - game.current_player  # Current player loses
                game.reason = "model_error"
                break
            
            # Self-correcting mechanism: Retry up to 5 times if the move is invalid
            max_retry_attempts = 5
            retry_count = 0
            move_interpreted = False
            
            while retry_count < max_retry_attempts:
                # Try to interpret ambiguous or incorrectly formatted moves using GPT-4.1-mini
                board_state = game.get_board_ascii()
                # Pass the current player (0 for white, 1 for black) to properly set active color
                interpreted_move = await interpret_move(raw_move, board_state, game.current_player)
                
                # If the move is valid, proceed
                if interpreted_move != "INVALID":
                    move_interpreted = True
                    break
                
                # Otherwise, log the invalid move and retry
                retry_count += 1
                logger.warning(f"Model {current_model} made an invalid move: '{raw_move}' that couldn't be interpreted (attempt {retry_count}/{max_retry_attempts})")
                
                # If we have more retries left, make another attempt
                if retry_count < max_retry_attempts:
                    # Send status update to the client about the retry
                    if not await safe_send_json({
                        "type": "status_update",
                        "data": {
                            "message": f"Retry attempt {retry_count}/{max_retry_attempts} for {current_model}..."
                        }
                    }):
                        # Client disconnected, clean up and return
                        game_manager.end_game(game_id)
                        return
                    
                    # Call the model again to get a new move
                    raw_move = await call_model(current_model, prompt)
                    
                    # Handle the case where we couldn't get any response from the model
                    if not raw_move:
                        logger.error(f"Failed to get any move from model {current_model} during retry {retry_count}")
                        # Try again if we have retries left, but don't count this as a retry if we got no response
                        continue
                    
                    logger.info(f"Retry {retry_count}: Received raw move '{raw_move}' from model {current_model}")
                
            # If we've exhausted all retries and still have an invalid move
            if not move_interpreted:
                logger.error(f"Model {current_model} failed to make a valid move after {max_retry_attempts} attempts")
                game.status = "finished"
                game.winner = 1 - game.current_player  # Current player loses
                game.reason = "invalid_notation"
                break
            
            # Try to apply the interpreted move with retry logic for invalid/illegal moves
            max_move_retry_attempts = 5  # Max attempts for parsing/applying the move
            move_retry_count = 0
            move_applied = False
            san_move = ""
            
            while move_retry_count < max_move_retry_attempts and not move_applied:
                try:
                    # First try to treat as UCI move
                    if len(interpreted_move) == 4 and interpreted_move[0] in "abcdefgh" and interpreted_move[1] in "12345678" and \
                       interpreted_move[2] in "abcdefgh" and interpreted_move[3] in "12345678":
                        uci_move = chess.Move.from_uci(interpreted_move)
                        if uci_move in game.board.legal_moves:
                            san_move = game.board.san(uci_move)
                            move = uci_move.uci()  # Normalize to UCI for internal processing
                            move_applied = True
                        else:
                            # UCI format but illegal move
                            raise ValueError(f"Illegal move: {interpreted_move}")
                    else:
                        # This is not UCI - attempt to parse as SAN properly
                        try:
                            # Note: python-chess requires calling from_san on the board object
                            san_move = interpreted_move  # Store the SAN format
                            uci_move = game.board.parse_san(interpreted_move)  # Correct method to use
                            move = uci_move.uci()  # Convert to UCI for internal processing
                            move_applied = True
                        except ValueError as e:
                            logger.error(f"Failed to parse '{interpreted_move}' as SAN: {e}")
                            raise
                
                except Exception as e:
                    # Handle specific errors with proper error messages
                    error_type = type(e).__name__
                    logger.error(f"Failed to convert move '{interpreted_move}': {error_type}: {str(e)}")
                    
                    # One more fallback attempt for some common errors
                    try:
                        # If we're here, the move might be in a non-standard format
                        # Try to handle it by performing a direct UCI extraction if it matches a pattern
                        if len(interpreted_move) >= 4:
                            # Look for sequences of file (a-h) and rank (1-8) that could form a move
                            possible_move = ""
                            for char in interpreted_move:
                                if char in "abcdefgh12345678":
                                    possible_move += char
                                    if len(possible_move) == 4:
                                        # Attempt to validate this as UCI
                                        from_sq = possible_move[0:2]
                                        to_sq = possible_move[2:4]
                                        if (from_sq[0] in "abcdefgh" and from_sq[1] in "12345678" and
                                            to_sq[0] in "abcdefgh" and to_sq[1] in "12345678"):
                                            # This looks like a valid UCI format
                                            test_move = chess.Move.from_uci(possible_move)
                                            if test_move in game.board.legal_moves:
                                                san_move = game.board.san(test_move)
                                                move = test_move.uci()
                                                logger.info(f"Recovered UCI move: {move} (SAN: {san_move})")
                                                move_applied = True
                                                break
                        
                        # If recovery failed, increment retry counter and try again if attempts remain
                        if not move_applied:
                            move_retry_count += 1
                            
                            if move_retry_count < max_move_retry_attempts:
                                logger.warning(f"Move parsing attempt {move_retry_count}/{max_move_retry_attempts} failed, retrying...")
                                
                                # Send status update to the client about the retry
                                client_connected = await safe_send_json({
                                    "type": "status_update",
                                    "data": {
                                        "message": f"Move parsing retry {move_retry_count}/{max_move_retry_attempts} for {current_model}..."
                                    }
                                })
                                
                                if not client_connected:
                                    # Client disconnected, clean up and return
                                    game_manager.end_game(game_id)
                                    return
                                
                                # Get a new move from the model
                                raw_move = await call_model(current_model, prompt)
                                
                                # Handle the case where we couldn't get any response from the model
                                if not raw_move:
                                    logger.error(f"Failed to get any move from model {current_model} during move parsing retry {move_retry_count}")
                                    continue
                                
                                logger.info(f"Move retry {move_retry_count}: Received raw move '{raw_move}' from model {current_model}")
                                
                                # Try to interpret the new move
                                interpreted_move = await interpret_move(raw_move, board_state, game.current_player)
                                
                                if interpreted_move == "INVALID":
                                    logger.warning(f"Interpreted move is still invalid during retry {move_retry_count}")
                                    continue
                            else:
                                # All retries exhausted
                                logger.error(f"Failed to parse/apply move after {max_move_retry_attempts} attempts")
                                raise ValueError(f"All move parsing retries failed for '{interpreted_move}'")
                        
                    except Exception as recovery_error:
                        # Recovery attempt also failed
                        logger.error(f"Recovery attempt failed: {str(recovery_error)}")
                        
                        # If we've tried enough times, game over
                        if move_retry_count >= max_move_retry_attempts - 1:
                            break
            
            # If we exhausted all retries and couldn't apply the move, end the game
            if not move_applied:
                game.status = "finished"
                game.winner = 1 - game.current_player
                game.reason = "invalid_notation"
                break
            
            # Additional debug logging
            logger.info(f"Processed move: Raw '{raw_move}' -> Interpreted '{interpreted_move}' -> UCI '{move}' -> SAN '{san_move}'")
            
            move_success = game.apply_move(move)
            
            # Get commentary for the move (asynchronously)
            if move_success:
                # Pass move history for better context
                commentary_task = await get_commentary(game.get_board_ascii(), san_move, game.moves_history)
                
                # Don't wait for the commentary - we'll check for it later
                # Just add a placeholder for now
                placeholder = "Analyzing move..."
                game.commentary.append(placeholder)
                
                # Store the task so we can check if it's ready later
                if not hasattr(game, 'pending_commentary'):
                    game.pending_commentary = {}
                
                # Store the commentary task along with the index in the commentary list
                game.pending_commentary[len(game.commentary) - 1] = commentary_task
            else:
                commentary = "Invalid move! This player loses the game."
                game.commentary.append(commentary)
            
            # Send updated game state with both raw and interpreted moves for transparency
            if not await safe_send_json({
                "type": "move_made",
                "data": {
                    "model": current_model,
                    "raw_move": raw_move,
                    "move": san_move,
                    "valid": move_success,
                    "interpreted": raw_move != interpreted_move,
                    "commentary": "Analyzing move..." if move_success else "Invalid move! This player loses the game."
                }
            }):
                # Client disconnected, clean up and return
                game_manager.end_game(game_id)
                return
            
            # Check if any pending commentary tasks have completed
            if hasattr(game, 'pending_commentary'):
                completed_indices = []
                for idx, task in list(game.pending_commentary.items()):
                    if task.done():
                        # Replace placeholder with actual commentary
                        try:
                            commentary = task.result()
                            game.commentary[idx] = commentary
                            completed_indices.append(idx)
                            
                            # Send the commentary update separately so UI can react
                            # If send fails, just continue - commentary is non-critical
                            # Make sure to include any current status info for display
                            await safe_send_json({
                                "type": "commentary_update",
                                "data": {
                                    "commentary": commentary,
                                    "index": idx,
                                    "model": current_model,
                                    "message": f"Waiting for {current_model} to make a move..."
                                }
                            })
                        except Exception as e:
                            logger.error(f"Error getting commentary result: {e}")
                            game.commentary[idx] = "Commentary unavailable due to an error."
                            completed_indices.append(idx)
                
                # Remove completed tasks
                for idx in completed_indices:
                    del game.pending_commentary[idx]
            
            # Send updated game state - continue even if this fails
            client_connected = await safe_send_json({
                "type": "game_state",
                "data": {
                    "game_id": game_id,
                    "model1": game.model1,
                    "model2": game.model2,
                    "current_player": game.current_player,
                    "board": game.board_to_unicode(),
                    "moves": game.moves_history,
                    "status": game.status,
                    "result": game.get_result(),
                    "commentary": game.commentary
                }
            })
            
            if not client_connected:
                # Client disconnected, clean up and return
                game_manager.end_game(game_id)
                return
            
            # Small delay to make game flow more readable
            await asyncio.sleep(1)
            
            # Check if there are any pending commentary tasks before sending next move request
            if hasattr(game, 'pending_commentary') and game.pending_commentary:
                # Wait a bit more to allow commentary to complete, but don't block the game forever
                # Just check a few times with small delays
                for _ in range(3):  # Try up to 3 times with short delays
                    # Check if any pending commentary tasks have completed
                    completed_indices = []
                    for idx, task in list(game.pending_commentary.items()):
                        if task.done():
                            try:
                                game.commentary[idx] = task.result()
                                completed_indices.append(idx)
                            except Exception as e:
                                logger.error(f"Error getting commentary result: {e}")
                                game.commentary[idx] = "Commentary unavailable due to an error."
                                completed_indices.append(idx)
                    
                    # Remove completed tasks
                    for idx in completed_indices:
                        del game.pending_commentary[idx]
                    
                    # If all tasks completed, break
                    if not game.pending_commentary:
                        break
                    
                    # Short delay
                    await asyncio.sleep(0.5)
            
                # If there are still pending tasks, let's update clients with latest commentary
                if game.pending_commentary:
                    # Send update but don't worry if it fails - commentary is non-critical
                    await safe_send_json({
                        "type": "game_state",
                        "data": {
                            "game_id": game_id,
                            "model1": game.model1,
                            "model2": game.model2,
                            "current_player": game.current_player,
                            "board": game.board_to_unicode(),
                            "moves": game.moves_history,
                            "status": game.status,
                            "result": game.get_result(),
                            "commentary": game.commentary
                        }
                    })
        
        # Game is over, send final state
        result = game.get_result()
        
        # Wait for any remaining commentary to complete
        if hasattr(game, 'pending_commentary') and game.pending_commentary:
            # For game over, we'll wait a bit longer for final commentary
            pending_tasks = list(game.pending_commentary.values())
            if pending_tasks:
                # Wait for all remaining tasks with a timeout
                done, _ = await asyncio.wait(pending_tasks, timeout=3.0)
                
                # Process completed tasks
                for idx, task in list(game.pending_commentary.items()):
                    if task.done():
                        try:
                            game.commentary[idx] = task.result()
                        except Exception as e:
                            logger.error(f"Error getting final commentary: {e}")
                            game.commentary[idx] = "Commentary unavailable due to an error."
        
        # Generate a friendly explanation of why the game ended
        try:
            friendly_reason = await generate_friendly_game_over_reason(
                result.get("reason", "unknown"), 
                result.get("winner_model", ""), 
                result.get("result", "unknown")
            )
        except Exception as e:
            logger.error(f"Error generating friendly game over reason: {e}")
            friendly_reason = f"Game ended due to {result.get('reason', 'unknown')}."
        
        # Send final game state - continue even if this fails
        await safe_send_json({
            "type": "game_over",
            "data": {
                "result": result,
                "board": game.board_to_unicode(),
                "commentary": game.commentary[-1] if game.commentary else "Game over!",
                "friendly_reason": friendly_reason
            }
        })
        
        # Save game result
        game_manager.end_game(game_id)
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for game {game_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
        # Try to send error message, but don't worry if this fails
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except:
            pass
    finally:
        # Ensure game is properly ended
        try:
            game_manager.end_game(game_id)
        except Exception as e:
            logger.error(f"Error ending game {game_id}: {e}")
            pass

# Run the application with Uvicorn when executing this file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)