# AI Chess Arena Development Plan

## Overview
Create a web application where AI models play chess against each other via OpenRouter API. Games will be automatically run with random models (or user-selected ones), with commentary from GPT-4o-mini.

## Components

### Backend (Python FastAPI)
1. **API Server**
   - Single file implementation with FastAPI
   - Model selection/randomization logic
   - Chess game state management
   - OpenRouter API integration
   - Game history storage

2. **Chess Game Logic**
   - Board representation and move validation
   - Game state tracking and turn management
   - Algebraic notation handling
   - Win/loss/draw detection

3. **Model Management**
   - Random model selection
   - User-specified model selection
   - Model leaderboard maintenance

4. **Commentary**
   - GPT-4o-mini integration for game commentary
   - Chess annotation symbols support

### Frontend (Single HTML file)
1. **UI Components**
   - Chess board visualization using Unicode symbols
   - Model selection dropdowns
   - Game status display
   - Move history
   - Leaderboard display
   - Commentary section

2. **Visual Effects**
   - Special messages and animations for notable events
   - Win celebration with confetti

3. **User Interactions**
   - Model selection
   - Starting new games
   - Viewing game history

## Implementation Plan

1. **Setup**
   - Create server.py with FastAPI
   - Create index.html with DaisyUI and TailwindCSS

2. **Backend Implementation**
   - Chess game logic
   - OpenRouter API integration
   - Model selection logic
   - Results storage

3. **Frontend Implementation**
   - Chess board representation
   - Model selection UI
   - Game status and commentary display
   - Visual effects

4. **Integration**
   - Connect frontend to backend API
   - Implement game flow
   - Add leaderboard functionality

5. **Testing**
   - Test API endpoints
   - Test game mechanics
   - Test model interactions

## Technical Details

- Use FastAPI for the backend server
- Unicode chess symbols for board visualization
- DaisyUI and TailwindCSS for styling
- OpenRouter API for AI model interactions
- Store results in results.txt
- HTML/CSS/JS for the frontend in a single index.html file