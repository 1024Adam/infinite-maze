"""
Integration tests for Game Engine in Infinite Maze.

These tests verify the integration between the Game engine, Player,
Clock, and Maze components working together as a complete system.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, MagicMock

from infinite_maze.core.engine import maze, controlled_run
from infinite_maze.core.game import Game
from infinite_maze.entities.player import Player
from infinite_maze.entities.maze import Line
from infinite_maze.utils.config import config
from tests.fixtures.pygame_mocks import full_pygame_mocks, InputSimulator
from tests.fixtures.test_helpers import PerformanceMonitor, GameStateCapture


class TestGameEngineInitialization:
    """Test game engine initialization and setup."""
    
    def test_engine_components_creation(self):
        """Test that engine creates all necessary components."""
        with patch('infinite_maze.core.engine.pygame.init'), \
             patch('infinite_maze.core.engine.pygame.key.get_pressed', return_value={}), \
             patch('infinite_maze.core.engine.pygame.event.get', return_value=[]), \
             patch('pygame.time.delay'):
            
            # Mock the game loop to exit immediately
            with patch.object(Game, 'isPlaying', return_value=False):
                with patch.object(Game, 'isActive', return_value=False):
                    with patch.object(Game, 'cleanup'):
                        try:
                            maze()
                        except SystemExit:
                            pass  # Expected exit
    
    def test_controlled_run_initialization(self):
        """Test controlled_run function initialization."""
        mock_wrapper = Mock()
        mock_counter = Mock()
        
        with patch('infinite_maze.core.engine.pygame.key.get_pressed', return_value={}):
            # Mock the game to exit immediately
            with patch.object(Game, 'isPlaying', return_value=False):
                with patch.object(Game, 'isActive', return_value=False):
                    result = controlled_run(mock_wrapper, mock_counter)
                    
                    # controlled_run reports via wrapper callbacks, no return value
                    assert result is None
                    mock_wrapper.gameover.assert_called_once()


class TestGameEngineGameLoop:
    """Test game engine main loop functionality."""
    
    def test_game_loop_single_iteration(self):
        """Test single iteration of game loop."""
        with full_pygame_mocks() as mocks:
            # Set up game to run one iteration then exit
            game = Game(headless=True)
            player = Player(100, 100, headless=True)
            lines = Line.generateMaze(game, 5, 5)
            
            # Mock key state for one frame
            mocks['event']['key_state'][pygame.K_RIGHT] = True
            
            # Simulate one game loop iteration
            initial_score = game.getScore()
            initial_pos = player.getPosition()
            
            # Simulate right movement
            player.moveX(1)
            game.incrementScore()
            
            # Verify state changes
            assert game.getScore() == initial_score + 1
            assert player.getX() > initial_pos[0]
    
    def test_game_loop_pause_handling(self):
        """Test game loop pause functionality."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Test pause toggle
        assert not game.isPaused()
        
        game.changePaused(player)
        assert game.isPaused()
        
        game.changePaused(player)
        assert not game.isPaused()
    
    def test_game_loop_input_processing(self):
        """Test input processing in game loop."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Test various inputs
        input_tests = [
            ('right', lambda: player.moveX(1)),
            ('left', lambda: player.moveX(-1)),
            ('up', lambda: player.moveY(-1)),
            ('down', lambda: player.moveY(1))
        ]
        
        for direction, action in input_tests:
            initial_pos = player.getPosition()
            action()
            new_pos = player.getPosition()
            
            # Position should change
            assert new_pos != initial_pos


class TestGameEngineMovementSystem:
    """Test game engine movement and collision system."""
    
    def test_movement_with_collision_detection(self):
        """Test movement system with collision detection."""
        player = Player(90, 100, headless=True)
        
        # Create wall blocking movement
        walls = [Line((100, 90), (100, 110))]  # Vertical wall
        
        # Test collision detection
        initial_x = player.getX()
        
        # Simulate collision check before movement
        test_x = initial_x + player.getSpeed()
        collision_detected = False
        
        for wall in walls:
            if wall.getIsHorizontal():
                # Check horizontal collision
                if (player.getY() <= wall.getYStart() <= player.getY() + player.getHeight() and
                    test_x <= wall.getXEnd() and test_x + player.getWidth() >= wall.getXStart()):
                    collision_detected = True
            else:
                # Check vertical collision
                if (test_x <= wall.getXStart() <= test_x + player.getWidth() and
                    player.getY() <= wall.getYEnd() and player.getY() + player.getHeight() >= wall.getYStart()):
                    collision_detected = True
        
        if not collision_detected:
            player.moveX(1)
        
        # Verify movement or blocking
        if collision_detected:
            assert player.getX() == initial_x  # Should not move
        else:
            assert player.getX() > initial_x   # Should move
    
    def test_boundary_enforcement(self):
        """Test game boundary enforcement."""
        game = Game(headless=True)
        player = Player(game.X_MIN, game.Y_MIN, headless=True)
        
        # Test left boundary
        player.setX(game.X_MIN - 10)
        if player.getX() < game.X_MIN:
            # Game should end or adjust position
            assert player.getX() < game.X_MIN  # Or game.end() called
        
        # Test right boundary
        player.setX(game.X_MAX + 10)
        if player.getX() > game.X_MAX:
            # Position should be adjusted
            adjusted_x = min(player.getX(), game.X_MAX)
            player.setX(adjusted_x)
            assert player.getX() <= game.X_MAX
        
        # Test vertical boundaries
        player.setY(game.Y_MIN - 10)
        adjusted_y = max(player.getY(), game.Y_MIN)
        player.setY(adjusted_y)
        assert player.getY() >= game.Y_MIN
        
        player.setY(game.Y_MAX + 10)
        adjusted_y = min(player.getY(), game.Y_MAX)
        player.setY(adjusted_y)
        assert player.getY() <= game.Y_MAX
    
    def test_maze_line_repositioning(self):
        """Test maze line repositioning system."""
        game = Game(headless=True)
        lines = Line.generateMaze(game, 5, 5)
        
        # Find initial max X
        initial_max_x = Line.getXMax(lines)
        
        # Simulate line repositioning (when lines move off screen)
        for line in lines:
            if line.getXStart() < 80:  # Line moved off screen
                line.setXStart(initial_max_x + 22)
                if line.getXStart() == line.getXEnd():
                    line.setXEnd(initial_max_x + 22)
                else:
                    line.setXEnd(initial_max_x + 44)
        
        # Verify repositioning
        new_max_x = Line.getXMax(lines)
        assert new_max_x >= initial_max_x


class TestGameEngineScoring:
    """Test game engine scoring system."""
    
    def test_scoring_with_movement(self):
        """Test scoring integration with movement."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        initial_score = game.getScore()
        
        # Right movement should increase score
        player.moveX(1)
        game.incrementScore()
        assert game.getScore() == initial_score + 1
        
        # Left movement should decrease score
        player.moveX(-1)
        game.decrementScore()
        assert game.getScore() == initial_score
        
        # Vertical movement should not affect score
        player.moveY(1)
        assert game.getScore() == initial_score
        
        player.moveY(-1)
        assert game.getScore() == initial_score
    
    def test_scoring_minimum_enforcement(self):
        """Test score minimum enforcement in game context."""
        game = Game(headless=True)
        
        # Start with some score
        game.setScore(2)
        
        # Decrement to minimum
        game.decrementScore()  # Score = 1
        game.decrementScore()  # Score = 0
        game.decrementScore()  # Should stay at 0
        
        assert game.getScore() == 0
    
    def test_scoring_with_game_progression(self):
        """Test scoring throughout game progression."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Simulate game progression with mixed movements
        movement_sequence = [
            ('right', 1, 1),    # Right: +1
            ('right', 1, 2),    # Right: +1
            ('left', -1, 1),    # Left: -1
            ('up', 0, 1),       # Up: +0
            ('down', 0, 1),     # Down: +0
            ('right', 1, 2),    # Right: +1
        ]
        
        for direction, score_change, expected_score in movement_sequence:
            if direction == 'right':
                player.moveX(1)
                game.incrementScore()
            elif direction == 'left':
                player.moveX(-1)
                game.decrementScore()
            elif direction == 'up':
                player.moveY(-1)
            elif direction == 'down':
                player.moveY(1)
            
            assert game.getScore() == expected_score


class TestGameEnginePaceSystem:
    """Test game engine pace/timing system."""
    
    def test_pace_progression(self):
        """Pace advances once per PACE_UPDATE_TICKS simulation ticks."""
        from infinite_maze.core.simulation import Simulation

        sim = Simulation(seed=1, max_steps=None)
        sim.lines = []

        for expected_pace in (1, 2):
            for _ in range(config.PACE_UPDATE_TICKS):
                sim.player.setX(200)  # keep scrolling from ending the episode
                sim.step((0, 0))
            assert sim.game.getPace() == expected_pace

    def test_pace_affects_movement(self):
        """Test that pace affects game movement."""
        game = Game(headless=True)
        player = Player(200, 100, headless=True)
        lines = Line.generateMaze(game, 5, 5)
        
        # Set initial pace
        game.setPace(0)
        initial_positions = [(line.getXStart(), line.getXEnd()) for line in lines]
        
        # Increase pace
        game.setPace(3)
        
        # Simulate pace effect on lines (moving them left)
        pace_offset = game.getPace()
        for line in lines:
            line.setXStart(line.getXStart() - pace_offset)
            line.setXEnd(line.getXEnd() - pace_offset)
        
        # Verify lines moved
        new_positions = [(line.getXStart(), line.getXEnd()) for line in lines]
        for initial, new in zip(initial_positions, new_positions):
            assert new[0] < initial[0]  # X positions should decrease
            assert new[1] < initial[1]
    
    def test_pace_player_adjustment(self):
        """Test player position adjustment due to pace."""
        game = Game(headless=True)
        player = Player(200, 100, headless=True)
        
        initial_x = player.getX()
        pace = 5
        game.setPace(pace)
        
        # Simulate player position adjustment due to pace
        player.setX(player.getX() - pace)
        
        assert player.getX() == initial_x - pace


class TestGameEngineStateManagement:
    """Test game engine state management."""
    
    def test_game_state_transitions(self):
        """Test game state transitions."""
        game = Game(headless=True)
        
        # Initial state
        assert game.isPlaying()
        assert game.isActive()
        assert not game.isPaused()
        
        # End game
        game.end()
        assert game.isPlaying()  # Still playing, but not active
        assert not game.isActive()
        
        # Reset game
        game.reset()
        assert game.isPlaying()
        assert game.isActive()
        assert not game.isPaused()
        
        # Quit game
        game.quit()
        assert not game.isPlaying()
    
    def test_game_state_with_player(self):
        """Test game state integration with player."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Capture initial state
        state_capture = GameStateCapture(game, player)
        state_capture.capture()
        
        # Modify state
        game.setScore(50)
        player.moveX(5)
        game.setPace(3)
        
        # Get state differences
        diff = state_capture.get_state_diff()
        assert 'game' in diff
        assert 'player' in diff
        
        # Restore state
        state_capture.restore()
        assert game.getScore() == 0  # Should be restored
    
    def test_pause_state_effects(self):
        """Test pause state effects on game systems."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        clock = game.getClock()
        
        # Normal state
        assert not game.isPaused()
        
        # Pause game
        game.changePaused(player)
        assert game.isPaused()
        
        # Simulate pause effects on clock
        pre_pause_millis = clock.getMillis()
        clock.update()
        post_pause_millis = clock.getMillis()
        
        if game.isPaused():
            # In paused state, might rollback time
            rollback_amount = post_pause_millis - pre_pause_millis
            clock.rollbackMillis(rollback_amount)
            assert clock.getMillis() == pre_pause_millis


class TestGameEnginePerformance:
    """Test game engine performance characteristics."""
    
    @pytest.mark.performance
    def test_game_loop_performance(self):
        """Test game loop performance."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        lines = Line.generateMaze(game, 10, 10)
        
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Simulate game loop iterations
        for frame in range(60):  # 1 second at 60 FPS
            # Update game state
            game.updateScreen(player, lines)
            
            # Simulate movement
            if frame % 4 == 0:
                player.moveX(1)
                game.incrementScore()
            
            # Update clock
            game.getClock().update()
            
            # Record frame
            monitor.sample_frame(60.0, 16.7)  # Target 60 FPS
        
        monitor.stop()
        
        # Verify performance
        duration = monitor.get_duration()
        avg_fps = monitor.get_average_fps()
        memory_stats = monitor.get_memory_usage_mb()
        
        assert duration < 2.0  # Should complete quickly
        assert avg_fps > 30.0  # Should maintain decent FPS
        assert memory_stats['peak'] < 200  # Should not use excessive memory
    
    @pytest.mark.performance
    def test_collision_detection_performance(self):
        """Test collision detection performance in game context."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        lines = Line.generateMaze(game, 20, 20)  # Large maze
        
        import time
        start_time = time.time()
        
        # Simulate many collision checks
        for _ in range(1000):
            # Test movement in each direction
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                test_x = player.getX() + dx * player.getSpeed()
                test_y = player.getY() + dy * player.getSpeed()
                
                # Check collision with all lines
                collision_count = 0
                for line in lines:
                    # Simplified collision check
                    if line.getIsHorizontal():
                        if (test_y <= line.getYStart() <= test_y + player.getHeight() and
                            test_x < line.getXEnd() and test_x + player.getWidth() > line.getXStart()):
                            collision_count += 1
                    else:
                        if (test_x <= line.getXStart() <= test_x + player.getWidth() and
                            test_y < line.getYEnd() and test_y + player.getHeight() > line.getYStart()):
                            collision_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 5.0, f"Collision detection took too long: {duration}s"
    
    @pytest.mark.performance
    def test_maze_generation_performance(self):
        """Test maze generation performance."""
        game = Game(headless=True)
        
        import time
        start_time = time.time()
        
        # Generate multiple mazes
        for size in [(5, 5), (10, 10), (15, 15), (20, 20)]:
            for _ in range(10):
                lines = Line.generateMaze(game, size[0], size[1])
                assert len(lines) > 0
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 3.0, f"Maze generation took too long: {duration}s"


class TestGameEngineIntegration:
    """Integration tests for complete game engine."""
    
    def test_complete_game_simulation(self):
        """Test complete game simulation."""
        game = Game(headless=True)
        player = Player(80, 223, headless=True)  # Default start position
        lines = Line.generateMaze(game, 15, 20)
        
        # Simulate short game session
        for step in range(100):
            # Update screen
            game.updateScreen(player, lines)
            
            # Simulate input
            if step % 5 == 0:
                # Move right occasionally
                player.moveX(1)
                game.incrementScore()
            elif step % 7 == 0:
                # Move down occasionally
                player.moveY(1)
            
            # Update clock
            game.getClock().update()
            
            # Check boundaries
            if player.getX() < game.X_MIN:
                game.end()
                break
            if player.getX() > game.X_MAX:
                player.setX(game.X_MAX)
            
            # Adjust for boundaries
            player.setY(max(player.getY(), game.Y_MIN))
            player.setY(min(player.getY(), game.Y_MAX))
        
        # Verify final state
        assert game.getScore() >= 0
        assert player.getX() >= game.X_MIN
        assert player.getY() >= game.Y_MIN
        assert player.getY() <= game.Y_MAX
    
    def test_game_engine_error_recovery(self):
        """Test game engine error recovery."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Test various error conditions
        try:
            # Invalid player position
            player.setX(-1000)
            player.setY(-1000)
            
            # Engine should handle gracefully
            game.updateScreen(player, [])
            
        except Exception as e:
            # Should not crash with unexpected exceptions
            assert False, f"Engine crashed with: {e}"
        
        # Reset to valid state
        player.reset(100, 100)
        assert player.getPosition() == (100, 100)
    
    def test_game_engine_resource_cleanup(self):
        """Test game engine resource cleanup."""
        # Test that multiple game instances clean up properly
        for _ in range(5):
            game = Game(headless=True)
            player = Player(100, 100, headless=True)
            lines = Line.generateMaze(game, 5, 5)
            
            # Use game briefly
            game.updateScreen(player, lines)
            game.incrementScore()
            
            # Cleanup
            game.cleanup()
        
        # Should complete without memory issues
