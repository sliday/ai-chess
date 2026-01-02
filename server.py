"""
AI Chess Arena - Backend Server
================================

A real-time chess platform where AI language models compete against each other.

Architecture:
    - FastAPI: Async HTTP server with WebSocket support
    - ChessGame: Manages individual game state (board, moves, commentary)
    - GameManager: Orchestrates games, WebSocket connections, and autonomous gameplay
    - Database: SQLite for game history and Elo leaderboard

Key Features:
    - Winner Retention: Previous winner stays for next game
    - Move Quality Detection: Analyzes commentary for move annotations
    - Smart Retries: Multi-tier system for handling invalid moves
    - Hallucination Detection: Fast-fail for moves from empty squares
    - Real-time Commentary: openai/gpt-5.1-chat provides game analysis
    - Elo Leaderboard: Tracks model performance over time
    - Statistics Dashboard: Platform-wide metrics with 30s caching

WebSocket Protocol:
    - game_state: Full board state updates
    - move_made: Move notifications with animation data
    - commentary_update: Async commentary from GPT-5.1
    - status_update: Game status messages
    - game_over: Final results with winner/reason

API Endpoints:
    - POST /games: Create new game (with optional winner retention)
    - GET /active_game: Get/create autonomous game
    - GET /stats: Platform statistics
    - GET /stats/hourly: 24-hour data for sparklines
    - GET /leaderboard: Model rankings by Elo

Created by: @stas_kulesh
Version: 1.5.0
Last Updated: 2025-11-28
"""

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
import database


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

# Initialize database
database.init_db()

# Initialize FastAPI app
app = FastAPI(title="AI Chess Arena")

# Security headers middleware (basic hardening suitable for this app)


# Read available models from models.txt and normalize IDs (remove leading slashes)
with open("models.txt", "r") as f:
    MODELS = [line.strip().lstrip('/') for line in f if line.strip()]

# Track which models support vision/image input
VISION_MODELS = set()

# Results file path
RESULTS_FILE = "results.txt"

# Function to fetch models from OpenRouter API
async def fetch_models_from_openrouter():
    """Fetch the latest models from OpenRouter API"""
    global VISION_MODELS
    try:
        logger.info("Fetching models from OpenRouter API...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                OPENROUTER_MODELS_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://ai-chess.example.com",
                    "X-Title": "AI Chess Arena"
                }
            )
            response.raise_for_status()
            data = response.json()

            # Extract model IDs and vision capability from the response
            models = []
            vision_models = set()
            if "data" in data:
                for model in data["data"]:
                    model_id = model.get("id", "").lstrip('/')
                    if model_id:
                        models.append(model_id)
                        # Check if model supports image input
                        architecture = model.get("architecture", {})
                        input_modalities = architecture.get("input_modalities", [])
                        if "image" in input_modalities:
                            vision_models.add(model_id)

            # Update global vision models set
            VISION_MODELS = vision_models
            logger.info(f"Fetched {len(models)} models from OpenRouter API ({len(vision_models)} support vision)")
            return models
    except Exception as e:
        logger.error(f"Error fetching models from OpenRouter: {e}")
        return None

# Configuration constants
MAX_MOVE_LENGTH = 10
MAX_RAW_MOVE_LENGTH = 50
MAX_MODEL_NAME_LENGTH = 100
MAX_MOVE_RETRIES = 5
MAX_API_RETRIES = 3
MOVE_LIMIT_THRESHOLD = 70
HTTP_TIMEOUT = 45.0
COMMENTARY_TIMEOUT = 30.0
GAME_DELAY = 1.0

# OpenRouter API key (get from environment variable)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set. API calls will fail.")

# Base URL for OpenRouter API
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_API_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_MODELS_COUNT_API_URL = "https://openrouter.ai/api/v1/models/count"

# Commentator model
COMMENTATOR_MODEL = "google/gemini-3-flash-preview"

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
    use_previous_result: bool = False  # Keep winner from previous game

# Class to represent a game result
class GameResult(BaseModel):
    model1: str
    model2: str
    winner: int  # 0 for model1, 1 for model2
    reason: Optional[str] = None
    timestamp: str

# Class to manage the chess game with full state tracking
class ChessGame:
    """
    Represents a chess game between two AI models.
    
    Manages the complete state of a chess game including board position,
    move history, player turns, and game completion detection.
    """
    
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
        self.commentary = []     # Commentary from the commentator model
        self.status = "in_progress"  # Game status (in_progress or finished)
        self.winner = None       # 0 for model1, 1 for model2, None for draw
        self.reason = None       # Reason for game ending (checkmate, stalemate, etc.)
        self.board_image_base64 = None  # Board image from frontend for vision models
        
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
            # Input validation
            if not validate_string_input(move_str, MAX_MOVE_LENGTH, "move_str"):
                self.status = "finished"
                self.winner = 1 - self.current_player
                self.reason = "invalid_notation"
                return False
                
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.moves_history.append(move_str)
                self.current_player = 1 - self.current_player  # Switch player
                return True
            else:
                logger.warning(f"Illegal move attempted: {move_str}")
                self.status = "finished"
                self.winner = 1 - self.current_player  # Current player loses
                self.reason = "illegal_move"
                return False
        except ValueError as e:
            logger.error(f"ValueError in apply_move: {e}")
            self.status = "finished"
            self.winner = 1 - self.current_player  # Current player loses
            self.reason = "invalid_notation"
            return False
        except Exception as e:
            logger.error(f"Unexpected error in apply_move: {e}")
            self.status = "finished"
            self.winner = 1 - self.current_player
            self.reason = "move_processing_error"
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
        loser_model = self.model2 if self.winner == 0 else self.model1
        return {
            "status": "finished",
            "result": "win",
            "winner": self.winner,
            "winner_model": winner_model,
            "loser_model": loser_model,
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
        
    def get_board_data(self) -> List[List[dict]]:
        """Convert chess board to representation with piece codes for SVG rendering
        
        Returns:
            A 2D array of dictionaries, each containing:
            - symbol: The piece code (e.g., 'wP', 'bK') or empty string
            - color: 'w' for white, 'b' for black, or empty for empty squares
        """
        # Map piece letters to their codes
        # White pieces: P, N, B, R, Q, K -> wP, wN, wB, wR, wQ, wK
        # Black pieces: p, n, b, r, q, k -> bP, bN, bB, bR, bQ, bK
        
        # Convert board to object format
        board_data = []
        
        # Get FEN representation
        fen = self.board.fen()
        fen_pieces = fen.split(' ')[0]
        fen_rows = fen_pieces.split('/')
        
        # Process each row from FEN
        for row_idx, fen_row in enumerate(fen_rows):
            board_row = []
            
            for char in fen_row:
                if char.isdigit():
                    # Empty squares
                    empty_count = int(char)
                    for _ in range(empty_count):
                        board_row.append({
                            "symbol": "",
                            "color": ""
                        })
                else:
                    # Piece
                    color = 'w' if char.isupper() else 'b'
                    piece_type = char.upper()
                    code = f"{color}{piece_type}"
                    board_row.append({
                        "symbol": code,
                        "color": color
                    })
            
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
        """Initialize the game manager"""
        # Dictionary of active games, keyed by game_id
        self.active_games: Dict[str, ChessGame] = {}
        # Active game loops
        self.game_tasks: Dict[str, asyncio.Task] = {}
        # WebSocket connections: game_id -> list of websockets
        self.connections: Dict[str, List[WebSocket]] = {}
        # Store result from previous game for smart pairing
        self.last_game_result: Optional[Dict] = None
        # Chat: websocket -> username mapping
        self.connection_usernames: Dict[WebSocket, str] = {}
        # Chat: rate limiting (websocket -> last message timestamp)
        self.last_chat_time: Dict[WebSocket, float] = {}
        # Track the current autonomous game ID
        self.autonomous_game_id: Optional[str] = None
        
    def save_result(self, result: GameResult, game: ChessGame):
        """Save game result to database"""
        database.save_game_result(
            game_id=f"game_{int(time.time())}", 
            model1=result.model1,
            model2=result.model2,
            winner=result.winner,
            reason=result.reason,
            moves=game.moves_history,
            fen=game.get_fen()
        )
    
    def create_game(self, model1: Optional[str] = None, model2: Optional[str] = None, use_previous_result: bool = False) -> str:
        """Create a new game with specified or random models

        Args:
            model1: First model (or 'random')
            model2: Second model (or 'random')
            use_previous_result: If True and there was a previous game with a winner,
                                keep the winner and randomize only the loser
        """
        # Input validation
        if model1 and not validate_string_input(model1, MAX_MODEL_NAME_LENGTH, "model1"):
            raise ValueError(f"Invalid model1: {model1}")
        if model2 and not validate_string_input(model2, MAX_MODEL_NAME_LENGTH, "model2"):
            raise ValueError(f"Invalid model2: {model2}")

        # Only allow models from our curated list when explicitly provided
        if model1 and model1 != "random":
            if normalize_model_id(model1) not in MODELS:
                raise HTTPException(status_code=400, detail="model1 is not in the allowed models list")
        if model2 and model2 != "random":
            if normalize_model_id(model2) not in MODELS:
                raise HTTPException(status_code=400, detail="model2 is not in the allowed models list")

        # Smart pairing: if previous game had a winner, keep winner and randomize loser
        if use_previous_result and self.last_game_result and self.last_game_result.get('winner') is not None:
            winner_index = self.last_game_result['winner']
            winner_model = self.last_game_result['winner_model']

            logger.info(f"Smart pairing: {winner_model} won last game, keeping them for next match")

            # Winner stays in the same position (model1 or model2)
            if winner_index == 0:
                model1 = winner_model
                # Randomize model2 (the loser)
                available_models = [m for m in MODELS if m != model1]
                model2 = random.choice(available_models)
            else:
                model2 = winner_model
                # Randomize model1 (the loser)
                available_models = [m for m in MODELS if m != model2]
                model1 = random.choice(available_models)
        else:
            # Full random pairing (for draws, initial games, or manual New Game)
            if not model1 or model1 == "random":
                model1 = random.choice(MODELS)

            if not model2 or model2 == "random":
                # Make sure we don't have model playing against itself
                available_models = [m for m in MODELS if m != model1]
                model2 = random.choice(available_models)
        
        # Ensure consistent model ID formatting (no leading slashes)
        normalized_model1 = normalize_model_id(model1)
        normalized_model2 = normalize_model_id(model2)
        
        game_id = f"game_{int(time.time())}_{random.randint(1000, 9999)}"
        self.active_games[game_id] = ChessGame(normalized_model1, normalized_model2)
        
        # Start the game loop in background
        self.game_tasks[game_id] = asyncio.create_task(self._game_loop(game_id))
        
        return game_id
    
    async def connect(self, websocket: WebSocket, game_id: str):
        """Connect a websocket to a game"""
        await websocket.accept()
        if game_id not in self.connections:
            self.connections[game_id] = []
        self.connections[game_id].append(websocket)
        # Assign anonymous username for chat
        username = f"Spectator_{random.randint(1000, 9999)}"
        self.connection_usernames[websocket] = username
        logger.info(f"Client connected to game {game_id} as {username}. Total clients: {len(self.connections[game_id])}")
        
    def disconnect(self, websocket: WebSocket, game_id: str):
        """Disconnect a websocket"""
        if game_id in self.connections:
            if websocket in self.connections[game_id]:
                self.connections[game_id].remove(websocket)
                logger.info(f"Client disconnected from game {game_id}")
        # Clean up chat metadata
        self.connection_usernames.pop(websocket, None)
        self.last_chat_time.pop(websocket, None)
    
    async def broadcast(self, game_id: str, message: dict):
        """Broadcast a message to all clients connected to a game"""
        if game_id in self.connections:
            disconnected = []
            for connection in self.connections[game_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Error broadcasting to client: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, game_id)

    async def _game_loop(self, game_id: str):
        """Main game loop running in background"""
        try:
            game = self.active_games[game_id]
            logger.info(f"Starting game loop for {game_id}: {game.model1} vs {game.model2}")
            
            # Game limits configuration
            game_start_time = asyncio.get_event_loop().time()
            GAME_TIMEOUT_SECONDS = 3600  # 1 hour per game
            MAX_MOVES = 70  # Maximum moves before evaluating position

            # Send initial state
            await self.broadcast(game_id, {
                "type": "game_state",
                "data": {
                    "game_id": game_id,
                    "model1": game.model1,
                    "model2": game.model2,
                    "current_player": game.current_player,
                    "board": game.get_board_data(),
                    "moves": game.moves_history,
                    "status": game.status,
                    "result": game.get_result(),
                    "commentary": game.commentary
                }
            })
            
            while not game.is_game_over():
                # Check for timeout
                elapsed_time = asyncio.get_event_loop().time() - game_start_time
                if elapsed_time > GAME_TIMEOUT_SECONDS:
                    logger.warning(f"Game {game_id} timed out after {elapsed_time:.1f} seconds")
                    game.status = "finished"
                    game.reason = "timeout"

                    # Count material to determine winner
                    material = game.calculate_material()
                    white_material = material['white']
                    black_material = material['black']

                    if white_material > black_material:
                        game.winner = 0  # White wins on material
                        logger.info(f"White wins on timeout by material: {white_material} vs {black_material}")
                    elif black_material > white_material:
                        game.winner = 1  # Black wins on material
                        logger.info(f"Black wins on timeout by material: {black_material} vs {white_material}")
                    else:
                        game.winner = None  # Draw if equal material
                        logger.info(f"Draw on timeout with equal material: {white_material} vs {black_material}")
                    break

                # Check for move limit
                move_count = len(game.moves_history)
                if move_count >= MAX_MOVES:
                    logger.warning(f"Game {game_id} reached move limit ({MAX_MOVES} moves)")
                    game.status = "finished"
                    game.reason = "move_limit"

                    # Count material to determine winner
                    material = game.calculate_material()
                    white_material = material['white']
                    black_material = material['black']

                    if white_material > black_material:
                        game.winner = 0  # White wins on material
                        logger.info(f"White wins on move limit by material: {white_material} vs {black_material}")
                    elif black_material > white_material:
                        game.winner = 1  # Black wins on material
                        logger.info(f"Black wins on move limit by material: {black_material} vs {white_material}")
                    else:
                        game.winner = None  # Draw if equal material
                        logger.info(f"Draw on move limit with equal material: {white_material} vs {black_material}")
                    break

                # Get the current player's model
                current_model = game.get_current_model()
                
                # Prepare the chess board state as a text representation for the model
                board_state = game.get_board_ascii()
                board_fen = game.board.fen()  # Add FEN for better position understanding

                # Format previous moves history for context
                moves_history_text = ""
                if game.moves_history:
                    moves_formatted = []
                    for i, move in enumerate(game.moves_history):
                        move_num = (i // 2) + 1
                        if i % 2 == 0:
                            moves_formatted.append(f"{move_num}. {move}")
                        else:
                            moves_formatted[-1] += f" {move}"
                    moves_history_text = "Previous moves:\n" + "\n".join(moves_formatted)

                prompt = f"""
You are playing as {"WHITE" if game.current_player == 0 else "BLACK"} in this chess game.
{"It is YOUR TURN to move now." if moves_history_text else "You are making the first move of the game."}

Current board position (YOUR pieces are {"UPPERCASE" if game.current_player == 0 else "lowercase"}):
{board_state}

FEN: {board_fen}

{moves_history_text}

IMPORTANT: Read the board carefully. Pieces have moved from their starting positions.
Do NOT suggest moves that replay the opening (e.g., e2e4 when e-pawn is already at e4).

No intro, no outro, only your next move in proper UCI format (e.g., "e2e4" for moving from e2 to e4). Your next move is:
"""
                
                # Send status update
                await self.broadcast(game_id, {
                    "type": "status_update",
                    "data": {
                        "message": f"Waiting for {current_model} to make a move...",
                        "commentary": game.commentary[-1] if game.commentary else ""
                    }
                })
                
                # Call model
                raw_move = await call_model(current_model, prompt)
                logger.info(f"Received raw move '{raw_move}' from model {current_model}")
                
                if not raw_move:
                    logger.error(f"Failed to get move from {current_model}")
                    game.status = "finished"
                    game.winner = 1 - game.current_player
                    game.reason = "model_error"
                    break
                
                # ============================================================
                # MOVE VALIDATION AND RETRY SYSTEM (Enhanced)
                # ============================================================
                # This section implements a two-tier retry strategy to handle AI model move generation:
                #
                # Tier 1: Move Interpretation Retries (max 5 attempts)
                # - Attempts to interpret the raw model output into valid UCI notation
                # - Uses interpret_move() which tries: direct UCI → SAN parsing → GPT interpretation
                # - **ENHANCED**: Now includes FEN notation + legal move examples in retries
                # - On failure, requests new move from model with different prompt phrasing
                #
                # Tier 2: API Call Retries (max 3 attempts per model call)
                # - Handles network failures, rate limits, model errors
                # - Progressively simplifies prompts and tries base model without suffix
                # - Falls back to openai/gpt-5.1-chat on final attempt
                #
                # Flow:
                # 1. Request move from model with primary prompt (includes FEN + position warning)
                # 2. Attempt to interpret raw_move with interpret_move() (hallucination detection active)
                # 3. If INVALID and retries remain:
                #    - Cycle to next fallback prompt (5 variations)
                #    - **NEW**: Include FEN, previous move warning, legal move examples
                #    - Request new move from model
                #    - Repeat until valid or max retries exhausted
                # 4. If exhausted, opponent wins by "invalid_notation"
                #
                # IMPROVEMENTS (Based on Log Analysis):
                # - Added FEN notation to help models understand position
                # - Show legal move COUNT (not examples - avoids move copying)
                # - Hallucination detection fast-fails moves from empty squares
                # - Position-aware warnings ("Don't replay e2e4 when pawn at e4")
                # - Explicit feedback: "Your previous move 'X' was INVALID"
                # - These changes target the ~5,677 "valid UCI but illegal" failures
                #
                # DESIGN DECISION: We deliberately DON'T show example legal moves in retries
                # to avoid "move copying" behavior where models lazily pick from examples
                # instead of strategically analyzing the position.
                # ============================================================

                # Self-correcting mechanism with varied prompts for different LLMs
                max_retry_attempts = 5
                retry_count = 0
                move_interpreted = False
                interpreted_move = "INVALID"

                # FALLBACK PROMPT VARIATIONS
                # Five progressively simplified prompt phrasings used on retries:
                # 1. "No explanation needed..." - Emphasizes brevity
                # 2. "Provide just the 4-character..." - Specifies exact format
                # 3. "Output only the move..." - Focuses on source+destination
                # 4. "UCI move only..." - Minimal instruction with example
                # 5. "Respond with exactly 4 characters..." - Extremely direct
                #
                # Strategy: Cycle through prompts to find phrasing that works for each model
                # Rationale: Different models trained on different instruction formats
                fallback_user_prompts = [
                    "No explanation needed. Respond with only a valid chess move in UCI format (e.g., 'e2e4'). Your move:",
                    "Provide just the 4-character UCI chess move notation. Example format: 'e2e4', 'g1f3'. Your move:",
                    "Output only the move in UCI notation (source square + destination square, like 'd2d4'). Nothing else:",
                    "UCI move only (4 characters: from-square to-square). Example: b1c3. Your move:",
                    "Respond with exactly 4 characters representing your chess move (e.g., 'e7e5'). Move:"
                ]

                while retry_count < max_retry_attempts:
                    board_state = game.get_board_ascii()
                    interpreted_move = await interpret_move(raw_move, board_state, game.current_player)

                    if interpreted_move != "INVALID":
                        move_interpreted = True
                        logger.info(f"✅ Move interpreted successfully: '{raw_move}' -> '{interpreted_move}'")
                        break

                    retry_count += 1
                    logger.warning(f"❌ Invalid move on attempt {retry_count}/{max_retry_attempts}: raw_move='{raw_move}', interpreted='{interpreted_move}'")
                    logger.warning(f"   Current player: {game.current_player}, Legal moves count: {len(list(game.board.legal_moves))}")

                    if retry_count < max_retry_attempts:
                        await self.broadcast(game_id, {
                            "type": "status_update",
                            "data": {"message": f"Retry attempt {retry_count}/{max_retry_attempts} for {current_model}..."}
                        })
                        logger.info(f"🔄 Requesting new move from {current_model} (retry {retry_count})")

                        # Use fallback prompt for retry (cycle through different prompts)
                        fallback_prompt_index = (retry_count - 1) % len(fallback_user_prompts)
                        fallback_user_msg = fallback_user_prompts[fallback_prompt_index]
                        logger.info(f"Using fallback prompt {fallback_prompt_index + 1}: {fallback_user_msg[:50]}...")

                        # Create simplified retry prompt with different phrasing + position hints
                        # NOTE: We don't show example legal moves to avoid move copying behavior
                        board_fen = game.board.fen()
                        legal_moves_count = len(list(game.board.legal_moves))

                        retry_prompt = f"""
You are playing as {"WHITE" if game.current_player == 0 else "BLACK"} in this chess game.
It is YOUR TURN to move now.

Current board position (YOUR pieces are {"UPPERCASE" if game.current_player == 0 else "lowercase"}):
{board_state}

FEN: {board_fen}

{moves_history_text}

WARNING: Your previous move '{raw_move}' was INVALID for this position.
- The board has changed since the opening - pieces have moved or been captured
- There are {legal_moves_count} legal moves available in this position
- Carefully analyze which pieces YOU control and where they can legally move

{fallback_user_msg}
"""

                        raw_move = await call_model(current_model, retry_prompt)
                        logger.info(f"📥 New raw move received: '{raw_move}'")
                        if not raw_move:
                            logger.error(f"Empty response from model on retry {retry_count}")
                            continue
                
                if not move_interpreted:
                    game.status = "finished"
                    game.winner = 1 - game.current_player
                    game.reason = "invalid_notation"
                    break
                
                # Apply move
                move_success = False
                san_move = ""
                uci_move_str = ""
                try:
                    # Try UCI first
                    if len(interpreted_move) == 4:
                        try:
                            uci_move = chess.Move.from_uci(interpreted_move)
                            if uci_move in game.board.legal_moves:
                                san_move = game.board.san(uci_move)
                                uci_move_str = interpreted_move
                                move_success = game.apply_move(interpreted_move)
                        except: pass

                    if not move_success:
                        # Try SAN
                        uci_move = game.board.parse_san(interpreted_move)
                        san_move = interpreted_move
                        uci_move_str = uci_move.uci()
                        move_success = game.apply_move(uci_move.uci())
                except Exception as e:
                    logger.error(f"Error applying move: {e}")

                if not move_success:
                    game.status = "finished"
                    game.winner = 1 - game.current_player
                    game.reason = "illegal_move"
                    break

                # Commentary (non-blocking) - pass board image if available for vision models
                commentary_task = await get_commentary(game.get_board_ascii(), san_move, game.moves_history, game.board_image_base64)
                commentary_index = len(game.commentary)
                game.commentary.append("Analyzing move...") # Placeholder

                # Send update
                await self.broadcast(game_id, {
                    "type": "move_made",
                    "data": {
                        "model": current_model,
                        "raw_move": raw_move,
                        "move": san_move,
                        "uci_move": uci_move_str,  # Add UCI format for animations
                        "valid": move_success,
                        "interpreted": raw_move != interpreted_move,
                        "commentary": "Analyzing move..."
                    }
                })

                # Process commentary in background (non-blocking)
                async def process_commentary():
                    try:
                        commentary = await commentary_task
                        game.commentary[commentary_index] = commentary
                        await self.broadcast(game_id, {
                            "type": "commentary_update",
                            "data": {
                                "commentary": commentary,
                                "index": commentary_index
                            }
                        })
                    except Exception as e:
                        logger.error(f"Commentary error: {e}")

                # Fire and forget - don't wait for commentary
                asyncio.create_task(process_commentary())
                
                # Send final state for this turn
                await self.broadcast(game_id, {
                    "type": "game_state",
                    "data": {
                        "game_id": game_id,
                        "model1": game.model1,
                        "model2": game.model2,
                        "current_player": game.current_player,
                        "board": game.get_board_data(),
                        "moves": game.moves_history,
                        "status": game.status,
                        "result": game.get_result(),
                        "commentary": game.commentary
                    }
                })
                
                # Small delay
                await asyncio.sleep(GAME_DELAY)
            
            # Game Over
            logger.info(f"Game {game_id} finished. Winner: {game.winner}")

            # Store result for smart pairing (for all games)
            result = game.get_result()
            self.last_game_result = {
                'winner': game.winner,
                'winner_model': game.model1 if game.winner == 0 else game.model2 if game.winner == 1 else None,
                'loser_model': game.model2 if game.winner == 0 else game.model1 if game.winner == 1 else None,
                'result': result.get('result'),
                'reason': game.reason
            }

            # Clear autonomous game ID if this was the autonomous game
            is_autonomous = (game_id == self.autonomous_game_id)
            if is_autonomous:
                self.autonomous_game_id = None

            self.end_game(game_id)

            # Send game over
            friendly_reason = await generate_friendly_game_over_reason(
                game.reason,
                game.model1 if game.winner == 0 else game.model2,
                "win" if game.winner is not None else "draw",
                game.model2 if game.winner == 0 else game.model1
            )

            await self.broadcast(game_id, {
                "type": "game_over",
                "data": {
                    "result": result,
                    "friendly_reason": friendly_reason
                }
            })

            # Start new autonomous game after delay (only if this was the autonomous game)
            if is_autonomous:
                await asyncio.sleep(10)
                self.start_autonomous_game()
            
        except Exception as e:
            logger.error(f"Error in game loop: {e}", exc_info=True)
            # Try to restart
            await asyncio.sleep(5)
            self.start_autonomous_game()
            
    def has_active_connections(self) -> bool:
        """Check if there are any active WebSocket connections"""
        total_connections = sum(len(conns) for conns in self.connections.values())
        return total_connections > 0

    def start_autonomous_game(self):
        """Start a new autonomous game if none is currently running and no users are connected"""
        # Cleanup old tasks
        done_tasks = [gid for gid, task in self.game_tasks.items() if task.done()]
        for gid in done_tasks:
            del self.game_tasks[gid]

        # Check if users are connected - if so, let them control game starts
        if self.has_active_connections():
            logger.info("Users are connected - skipping autonomous game start")
            return

        # Only start a new autonomous game if we don't already have one
        if self.autonomous_game_id is None or self.autonomous_game_id not in self.game_tasks:
            logger.info("Starting new autonomous game (no active connections)...")
            # Use smart pairing: winner stays, loser is replaced
            game_id = self.create_game(use_previous_result=True)
            # Mark this as the autonomous game
            self.autonomous_game_id = game_id
            logger.info(f"Autonomous game ID set to: {game_id}")

# Initialize game manager

    
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
                self.save_result(result, game)
            
            # Remove game
            del self.active_games[game_id]
    
    def get_leaderboard(self, min_games: int = 0) -> List[Dict[str, Any]]:
        """Generate a leaderboard based on game results"""
        return database.get_leaderboard_stats(min_games)

    def get_statistics(self, period: str = "daily") -> Dict[str, Any]:
        """Get daily or weekly statistics

        Args:
            period: 'daily' or 'weekly'

        Returns:
            Dictionary with stats for daily highlights and weekly trends
        """
        return database.get_statistics(period)

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

# Utility functions for common operations
def validate_string_input(value: str, max_length: int, field_name: str) -> bool:
    """
    Validate string input with comprehensive checks.
    
    Args:
        value: The string value to validate
        max_length: Maximum allowed length for the string
        field_name: Name of the field for error logging
        
    Returns:
        True if valid, False otherwise
    """
    if not value or not isinstance(value, str):
        logger.warning(f"Invalid {field_name} input: {value}")
        return False
    
    if len(value.strip()) > max_length:
        logger.warning(f"{field_name} too long: {len(value)} characters")
        return False
    
    return True

def safe_file_operation(file_path: str, operation: str, func, *args, **kwargs):
    """Safely perform file operations with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error during {operation} on {file_path}: {e}")
        return None

def normalize_model_id(model_id: str) -> str:
    """Normalize model ID by removing leading slashes"""
    if not model_id:
        return ""
    return model_id.lstrip('/')

# Initialize game manager
game_manager = GameManager()

# Function to call OpenRouter API
async def interpret_move(raw_move: str, board_state: str, current_player: int = 0) -> str:
    """
    MOVE INTERPRETATION PIPELINE (Enhanced with Hallucination Detection)

    Converts raw model output into valid UCI moves through 4-step cascade with anti-hallucination checks.

    Step 1: Input Validation & UCI Direct Check
    - Checks string length (max 50 chars)
    - Validates current_player (0=white, 1=black)
    - Validates UCI format and checks legality
    - **NEW**: Hallucination Detection - Fast-fails moves from empty squares
      * Detects when models output e2e4 but e2 is empty (piece already moved)
      * Prevents wasteful retries for position-unaware model outputs
      * Logs clear diagnostic: "HALLUCINATION DETECTED"

    Step 2: Preprocessing & Cleanup
    - Extracts moves from code blocks (```e2e4```)
    - Strips common phrases ("Here is your move: e2e4")
    - Handles multiline responses (takes first line)
    - Removes piece prefixes (pe2e4 → e2e4)

    Step 3: Direct Parsing (deterministic, no API)
    a) Exact UCI format check (e.g., e2e4, g1f3)
    b) Pawn shorthand expansion (e4 → e2e4 or e7e5)
    c) SAN to UCI conversion (Nf3 → g1f3) - **36+ successful conversions in production**
    d) Format-valid but illegal move detection - skips expensive API call

    Step 4: GPT Interpretation (last resort)
    - Calls openai/gpt-5.1-chat to interpret ambiguous moves
    - Validates interpreted move is legal on current board
    - Returns INVALID if interpretation fails or is illegal

    Args:
        raw_move: The raw move string from the model
        board_state: The current board state (ASCII or FEN)
        current_player: Which player is making the move (0 for white, 1 for black)

    Returns:
        Valid UCI move string or "INVALID"

    Critical Validations:
    - All moves validated against current_board.legal_moves
    - Empty square detection prevents ~40% of retry failures
    - SAN→UCI conversion successful for models that output algebraic notation

    Log Analysis Insights (from 1.1M+ line production logs):
    - 5,677 "valid UCI but illegal on board" warnings (primarily model hallucinations)
    - Models like neversleep/noromaid-20b consistently output starting position moves
    - Retry success rate: ~13% (750 successes, most via SAN conversion)
    - Primary failure: Models don't read board state, hallucinate initial position
    """
    # Input validation
    if not validate_string_input(raw_move, MAX_RAW_MOVE_LENGTH, "raw_move"):
        return "INVALID"
    
    if not isinstance(current_player, int) or current_player not in [0, 1]:
        logger.warning(f"Invalid current_player: {current_player}")
        return "INVALID"
    
    # Clean up the raw move (remove extra spaces, punctuation, newlines)
    raw_move = raw_move.strip().replace('.', '').replace(',', '').replace(':', '')
    
    # ENHANCED: Try to extract a valid UCI move from unstructured responses
    # Look for patterns like "Here is your move: e2e4" or "My move is e2e4"
    import re
    
    # Remove code blocks if present (```e2e4``` or ```chess e2e4```)
    if '```' in raw_move:
        # Extract content between code blocks
        code_block_pattern = r'```(?:chess|uci|move)?\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*```'
        match = re.search(code_block_pattern, raw_move.lower())
        if match:
            extracted_move = match.group(1)
            logger.info(f"Extracted move '{extracted_move}' from code block: '{raw_move}'")
            raw_move = extracted_move
        else:
            # Just remove the code block markers
            raw_move = raw_move.replace('```', '').strip()
            logger.info(f"Removed code block markers: '{raw_move}'")
    
    # Pattern 1: Find UCI move after common phrases
    phrases_pattern = r'(?:here is|my move is|i move|i play|move:|the move is|i choose|i suggest|i recommend)[\s:]*([a-h][1-8][a-h][1-8][qrbn]?)'
    match = re.search(phrases_pattern, raw_move.lower())
    if match:
        extracted_move = match.group(1)
        logger.info(f"Extracted move '{extracted_move}' from response: '{raw_move}'")
        raw_move = extracted_move
    
    # Pattern 2: Find any UCI move pattern in the response (4-5 chars: e2e4 or e7e8q)
    if not match:
        uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
        matches = re.findall(uci_pattern, raw_move.lower())
        if matches:
            # Take the first valid-looking UCI move
            extracted_move = matches[0]
            logger.info(f"Extracted UCI pattern '{extracted_move}' from response: '{raw_move}'")
            raw_move = extracted_move
    
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

        # If it looks like an ASCII board, parse it properly
        if ascii_board_pattern:
            try:
                # Parse ASCII board format (e.g., "r n b q k b n r\np p p p p p p p\n...")
                lines = [l.strip() for l in board_state.strip().split('\n') if l.strip()]
                # Filter to only lines that look like board rows (contain pieces/dots)
                board_lines = [l for l in lines if any(c in l for c in 'rnbqkpRNBQKP.')]

                if len(board_lines) == 8:
                    # Convert ASCII to FEN notation
                    fen_ranks = []
                    for line in board_lines:
                        # Split by spaces and convert to FEN rank notation
                        pieces = line.split()
                        if len(pieces) != 8:
                            raise ValueError(f"Invalid board row: {line}")

                        fen_rank = ""
                        empty_count = 0
                        for piece in pieces:
                            if piece == '.':
                                empty_count += 1
                            else:
                                if empty_count > 0:
                                    fen_rank += str(empty_count)
                                    empty_count = 0
                                fen_rank += piece

                        # Add any trailing empty squares
                        if empty_count > 0:
                            fen_rank += str(empty_count)

                        fen_ranks.append(fen_rank)

                    # Combine ranks with slashes
                    fen_board = '/'.join(fen_ranks)
                    # Add game state components
                    full_fen = f"{fen_board} {active_color} KQkq - 0 1"

                    # Create board from parsed FEN
                    current_board = chess.Board(full_fen)
                    logger.info(f"Successfully parsed ASCII board to FEN with active color: {active_color}")
                else:
                    logger.warning(f"ASCII board has {len(board_lines)} rows instead of 8, using default")
            except Exception as e:
                logger.warning(f"Could not parse ASCII board: {e} - using default board")
                # Keep the default board we already created
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
            else:
                # Step 1b: Detect hallucinated starting position moves
                # Models often output moves from initial position (e.g., e2e4) even when pieces moved
                from_square = chess.parse_square(raw_move[:2])
                piece = current_board.piece_at(from_square)

                if piece is None:
                    logger.warning(f"🚨 HALLUCINATION DETECTED: Move '{raw_move}' tries to move from empty square {raw_move[:2]}")
                    logger.warning(f"   This suggests model is imagining starting position instead of reading current board")
                    return "INVALID"  # Fast-fail without expensive API call

                # Check if it's a typical opening move from starting rank but piece isn't there
                starting_ranks = {'2', '7'}  # White and black starting pawn ranks
                if raw_move[1] in starting_ranks and piece.piece_type == chess.PAWN:
                    # Pawn move from starting rank - verify it hasn't moved yet
                    if raw_move[1] == '2' and current_player == 0:  # White pawn
                        pass  # This might be valid first move
                    elif raw_move[1] == '7' and current_player == 1:  # Black pawn
                        pass  # This might be valid first move
                    else:
                        logger.warning(f"⚠️ Suspicious: Move '{raw_move}' looks like opening move but may be replaying past moves")

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
        pass  # Continue to next checks if direct parsing fails

    # Step 3d: Check if move format is valid UCI but just illegal on current board
    # Don't waste API calls on moves that are formatted correctly but tactically illegal
    if len(raw_move) == 4 and all([
        raw_move[0] in "abcdefgh", raw_move[1] in "12345678",
        raw_move[2] in "abcdefgh", raw_move[3] in "12345678"
    ]):
        # Valid UCI format but not legal on this board
        logger.warning(f"Move '{raw_move}' is valid UCI format but illegal on current board")
        legal_moves_sample = [m.uci() for m in list(current_board.legal_moves)[:5]]
        logger.warning(f"Legal moves sample: {legal_moves_sample}...")
        return "INVALID"  # Skip expensive API call

    # Step 4: If all direct parsing attempts fail, use GPT-5.1-chat to interpret
    logger.info(f"Attempting to interpret ambiguous move with API: '{raw_move}'")
    
    # Get proper FEN from board
    current_fen = current_board.fen()
    
    # Normalize model ID: remove leading slash if present
    interpreter_model = "openai/gpt-5.1-chat"  # Use better model for chess move interpretation
    
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
        "max_tokens": 50  # USER APPROVED: Safe buffer for OpenRouter proxy overhead (Azure requires >=16)
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

                # If the interpretation is empty, log and return invalid
                if not interpretation:
                    logger.warning(f"Interpreter returned empty string for '{raw_move}'")
                    logger.warning(f"Possible rate limit or model refusal")
                    return "INVALID"

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
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} interpreting move '{raw_move}'")
            logger.error(f"Response body: {e.response.text}")
            if "max_tokens" in e.response.text or "max_output_tokens" in e.response.text:
                logger.error(f"⚠️ TOKEN PARAMETER ERROR - Check max_tokens configuration")
            return "INVALID"

        except Exception as e:
            logger.error(f"Error interpreting move '{raw_move}': {e}")
            return "INVALID"

async def call_model(model: str, prompt: str, max_retries: int = MAX_API_RETRIES) -> str:
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
        model_id = "openai/gpt-5.1-chat"
    
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
    
    # Enhanced payload with strict system message to minimize unstructured responses
    system_message = """You are a chess engine. You MUST respond with ONLY a chess move in UCI notation.

CRITICAL RULES:
- Output ONLY the move (e.g., e2e4, g1f3, e7e8q)
- NO explanations, NO text before or after
- NO phrases like "Here is your move:" or "My move is:"
- JUST the 4-5 character UCI move
- Example valid responses: e2e4, g1f3, a7a8q

Your move:"""
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
                current_model = "openai/gpt-5.1-chat"
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
                    if current_model != "openai/gpt-5.1-chat" and (attempt < max_retries):
                        current_model = "openai/gpt-5.1-chat"
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
                    
                    # ENHANCED DEBUGGING: Log the actual raw response content
                    logger.info(f"Raw model response from {current_model}: '{move}'")
                    
                    # Log if response contains common problematic patterns
                    if "```" in move:
                        logger.warning(f"Model response contains code blocks (```): '{move}'")
                    if any(phrase in move.lower() for phrase in ["here is", "my move", "i move", "i play"]):
                        logger.warning(f"Model response contains conversational text: '{move}'")
                    if len(move) > 10:
                        logger.warning(f"Model response is longer than expected ({len(move)} chars): '{move}'")
                    
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

async def get_commentary(board_state: str, last_move: str, moves_history: list = None, board_image_base64: str = None) -> str:
    """Get chess commentary asynchronously without blocking the game"""
    # Create a task for the commentary but don't await it immediately
    return asyncio.create_task(_fetch_commentary(board_state, last_move, moves_history, board_image_base64))

async def generate_friendly_game_over_reason(reason: str, winner_model: str, result_type: str, loser_model: str = "") -> str:
    """Generate a user-friendly explanation for why the game ended
    
    Args:
        reason: Technical reason the game ended (e.g., 'invalid_notation', 'checkmate')
        winner_model: The model that won (if any)
        result_type: Type of result ('win', 'draw', etc.)
        loser_model: The model that lost (if any)
        
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
        explanation_prompt = f"Explain why {loser_model} failed to make a valid chess move (friendly explanation for non-experts)."
    elif reason == "illegal_move":
        explanation_prompt = f"Explain why {loser_model} attempted an illegal chess move (friendly explanation for non-experts)."
    elif reason == "checkmate":
        explanation_prompt = "Describe the victory by checkmate in fun, simple terms."
    elif reason == "reached_move_limit":
        explanation_prompt = "Explain why the game ended after reaching the 70-move limit and how a winner was determined based on material."
    elif reason == "timeout":
        if result_type == "draw":
            explanation_prompt = "Explain that the 20-minute game timer expired, and since both sides had equal material (pieces), the game ended in a draw."
        else:
            explanation_prompt = f"Explain that the 20-minute game timer expired, and {winner_model} won by having more valuable pieces on the board (material advantage)."
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
- Loser: {loser_model if result_type == 'win' and loser_model else 'None (Draw)'}
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

async def _fetch_commentary(board_state: str, last_move: str, moves_history: list = None, board_image_base64: str = None) -> str:
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

    # Check if model supports vision and we have an image
    use_vision = board_image_base64 and commentator_model in VISION_MODELS

    # Build user message content
    text_prompt = f"""Provide a brief analysis of the latest chess move and current position.

Current board state:
{board_state}{moves_context}

Last move: {last_move}

Use these chess annotation symbols when appropriate:
!! (brilliant move), ! (good move), ?! (questionable move), ? (mistake), ?? (blunder), !? (interesting move),
⊕ (with an attack), ∞ (unclear position), = (equal position),
± (White has a slight advantage), ∓ (Black has a slight advantage),
+− (White has a decisive advantage), −+ (Black has a decisive advantage)
"""

    if use_vision:
        # Multi-modal message with image
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{board_image_base64}"
                }
            },
            {
                "type": "text",
                "text": text_prompt
            }
        ]
        logger.info(f"Requesting commentary from {commentator_model} with board image")
    else:
        # Text-only message
        user_content = text_prompt
        if board_image_base64 and commentator_model not in VISION_MODELS:
            logger.debug(f"Model {commentator_model} doesn't support vision, using text-only")

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
                "content": user_content
            }
        ],
        "max_tokens": 100
    }

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

@app.get("/models/count")
async def get_models_count():
    """Get the count of available models from OpenRouter API"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                OPENROUTER_MODELS_COUNT_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://ai-chess.example.com",
                    "X-Title": "AI Chess Arena"
                }
            )
            response.raise_for_status()
            data = response.json()

            # Return the count from the API response
            count = data.get("count", len(MODELS))
            return {"count": count}
    except Exception as e:
        logger.error(f"Error fetching model count from OpenRouter: {e}")
        # Fallback to local count
        return {"count": len(MODELS)}

@app.post("/refresh-models")
async def refresh_models():
    """Refresh models from OpenRouter API"""
    global MODELS

    new_models = await fetch_models_from_openrouter()

    if new_models is None:
        raise HTTPException(status_code=500, detail="Failed to fetch models from OpenRouter API")

    if len(new_models) == 0:
        raise HTTPException(status_code=500, detail="No models returned from OpenRouter API")

    # Update the global MODELS list
    MODELS = new_models

    # Save to models.txt as cache
    try:
        with open("models.txt", "w") as f:
            for model in MODELS:
                f.write(f"{model}\n")
        logger.info(f"Saved {len(MODELS)} models to models.txt")
    except Exception as e:
        logger.error(f"Error saving models to file: {e}")

    return {
        "success": True,
        "model_count": len(MODELS),
        "message": f"Successfully refreshed {len(MODELS)} models from OpenRouter API"
    }

@app.post("/games")
async def create_game(request: ModelSelectionRequest):
    """Create a new game

    Args:
        request: Model selection parameters including optional winner retention
    """
    game_id = game_manager.create_game(request.model1, request.model2, request.use_previous_result)
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

@app.get("/stats")
async def get_stats(period: str = "daily"):
    """Get daily or weekly statistics

    Args:
        period: 'daily' or 'weekly' (default: daily)
    """
    return game_manager.get_statistics(period)

@app.get("/stats/hourly")
async def get_hourly_stats():
    """Get hourly time-series data for sparklines (last 24 hours)"""
    return database.get_hourly_stats()

@app.get("/leaderboard")
async def get_leaderboard(min_games: int = 3):
    """Get model leaderboard

    Args:
        min_games: Minimum number of games required to appear on leaderboard (default: 3)
    """
    return {"leaderboard": game_manager.get_leaderboard(min_games)}

@app.get("/active_game")
async def get_active_game():
    """Get the ID of the currently active autonomous game"""
    # Find the first active game
    if game_manager.active_games:
        game_id = list(game_manager.active_games.keys())[0]
        return {"game_id": game_id}

    # If no game, start one
    game_id = game_manager.create_game()
    return {"game_id": game_id}

@app.get("/viewers_count")
async def get_viewers_count():
    """Get the count of currently connected viewers (WebSocket connections)"""
    total_viewers = sum(len(connections) for connections in game_manager.connections.values())
    return {"count": total_viewers}

@app.get("/game_status/{game_id}")
async def get_game_status(game_id: str):
    """Check if a game exists and get its status"""
    if game_id in game_manager.active_games:
        game = game_manager.active_games[game_id]
        return {
            "exists": True,
            "status": game.status,
            "model1": game.model1,
            "model2": game.model2
        }
    return {"exists": False}

@app.on_event("startup")
async def startup_event():
    """Start the autonomous game loop on startup"""
    # Wait a bit for server to start
    asyncio.create_task(delayed_start())

async def delayed_start():
    await asyncio.sleep(2)
    # Fetch models to populate VISION_MODELS set
    await fetch_models_from_openrouter()
    logger.info(f"Initialized with {len(VISION_MODELS)} vision-capable models")
    game_manager.start_autonomous_game()

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket connection for real-time game updates"""
    await game_manager.connect(websocket, game_id)
    
    try:
        # Send current state immediately
        if game_id in game_manager.active_games:
            game = game_manager.active_games[game_id]
            await websocket.send_json({
                "type": "game_state",
                "data": {
                    "game_id": game_id,
                    "model1": game.model1,
                    "model2": game.model2,
                    "current_player": game.current_player,
                    "board": game.get_board_data(),
                    "moves": game.moves_history,
                    "status": game.status,
                    "result": game.get_result(),
                    "commentary": game.commentary
                }
            })

        # Send username to client (for chat)
        username = game_manager.connection_usernames.get(websocket, "Spectator")
        await websocket.send_json({"type": "welcome", "username": username})

        # Send recent chat messages from database
        chat_history = database.get_chat_messages(game_id, limit=30)
        if chat_history:
            await websocket.send_json({
                "type": "chat_history",
                "messages": chat_history
            })

        # Keep connection open and handle incoming messages
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                # Handle board image upload from client
                if data.get("type") == "board_image" and game_id in game_manager.active_games:
                    game = game_manager.active_games[game_id]
                    game.board_image_base64 = data.get("image")
                    logger.debug(f"Received board image for game {game_id}")

                # Handle chat message
                elif data.get("type") == "chat_message":
                    # Rate limiting: 1 message per second
                    now = time.time()
                    last = game_manager.last_chat_time.get(websocket, 0)
                    if now - last < 1.0:
                        continue  # Silently ignore rapid messages

                    username = game_manager.connection_usernames.get(websocket, "Anonymous")
                    text = data.get("text", "")[:200]  # Max 200 chars

                    if text.strip():
                        game_manager.last_chat_time[websocket] = now
                        msg = {
                            "type": "chat_message",
                            "username": username,
                            "text": text,
                            "timestamp": now
                        }
                        # Save to database
                        database.save_chat_message(game_id, username, text, now)
                        # Broadcast to all viewers
                        await game_manager.broadcast(game_id, msg)

            except json.JSONDecodeError:
                pass  # Ignore non-JSON messages (keep-alive, etc.)

    except WebSocketDisconnect:
        game_manager.disconnect(websocket, game_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        game_manager.disconnect(websocket, game_id)

# Run the application with Uvicorn when executing this file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)