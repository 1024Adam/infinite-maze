"""
Unit tests for the Game class in Infinite Maze.

These tests verify the functionality of the Game class including
game state management, scoring, timing, display handling, and configuration.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, MagicMock

from infinite_maze.core.game import Game
from infinite_maze.core.clock import Clock
from tests.fixtures.pygame_mocks import full_pygame_mocks, headless_pygame_mocks
from tests.fixtures.test_helpers import temporary_game_state


class TestGameInitialization:
    """Test Game class initialization."""
    
    def test_game_init_headless_mode(self):
        """Test game initialization in headless mode."""
        game = Game(headless=True)
        
        # Basic state checks
        assert game.getScore() == 0
        assert game.getPace() == 0
        assert not game.isPaused()
        assert game.isActive()
        assert game.isPlaying()
        
        # Headless mode should not initialize display components
        assert game.screen is None
        assert game.font is None
        assert game.icon is None
    
    def test_game_init_with_display(self):
        """Test game initialization with display mode."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            
            # Verify pygame initialization calls
            assert mocks['display']['set_mode'].called
            assert mocks['display']['set_caption'].called
            assert mocks['display']['set_icon'].called
            
            # Game should be properly initialized
            assert game.getScore() == 0
            assert game.getPace() == 0
            assert not game.isPaused()
    
    def test_game_constants(self):
        """Test game constants are properly set."""
        game = Game(headless=True)
        
        # Check that display constants are reasonable
        assert game.WIDTH > 0
        assert game.HEIGHT > 0
        assert game.X_MIN >= 0
        assert game.Y_MIN >= 0
        assert game.X_MAX > game.X_MIN
        assert game.Y_MAX > game.Y_MIN
        
        # Check score increment
        assert game.SCORE_INCREMENT == 1


class TestGameScoring:
    """Test Game scoring functionality."""
    
    def test_initial_score(self):
        """Test initial score is zero."""
        game = Game(headless=True)
        assert game.getScore() == 0
    
    def test_increment_score(self):
        """Test score increment functionality."""
        game = Game(headless=True)
        initial_score = game.getScore()
        
        game.incrementScore()
        
        assert game.getScore() == initial_score + game.SCORE_INCREMENT
    
    def test_decrement_score(self):
        """Test score decrement functionality."""
        game = Game(headless=True)
        
        # First increment to have a positive score
        game.incrementScore()
        game.incrementScore()
        current_score = game.getScore()
        
        game.decrementScore()
        
        assert game.getScore() == current_score - game.SCORE_INCREMENT
    
    def test_decrement_score_at_zero(self):
        """Test that score cannot go below zero."""
        game = Game(headless=True)
        
        # Try to decrement from zero
        game.decrementScore()
        
        assert game.getScore() == 0
    
    def test_decrement_score_minimum_enforcement(self):
        """Test minimum score enforcement with repeated decrements."""
        game = Game(headless=True)
        
        # Increment once, then decrement multiple times
        game.incrementScore()
        game.decrementScore()
        game.decrementScore()
        game.decrementScore()
        
        assert game.getScore() == 0
    
    def test_update_score(self):
        """Test updateScore method with positive and negative values."""
        game = Game(headless=True)
        
        # Test positive update
        game.updateScore(5)
        assert game.getScore() == 5
        
        # Test negative update
        game.updateScore(-2)
        assert game.getScore() == 3
        
        # Test zero update
        game.updateScore(0)
        assert game.getScore() == 3
    
    def test_set_score(self):
        """Test setScore method."""
        game = Game(headless=True)
        
        game.setScore(42)
        assert game.getScore() == 42
        
        game.setScore(0)
        assert game.getScore() == 0
        
        # Test setting negative score (should be allowed by setScore)
        game.setScore(-10)
        assert game.getScore() == -10
    
    def test_score_sequence(self):
        """Test a sequence of score operations."""
        game = Game(headless=True)
        
        operations = [
            ('increment', 1),
            ('increment', 2),
            ('increment', 3),
            ('decrement', 2),
            ('update', 5, 7),  # updateScore(5) should result in 7
            ('set', 10, 10)    # setScore(10) should result in 10
        ]
        
        for operation in operations:
            if operation[0] == 'increment':
                game.incrementScore()
                assert game.getScore() == operation[1]
            elif operation[0] == 'decrement':
                game.decrementScore()
                assert game.getScore() == operation[1]
            elif operation[0] == 'update':
                game.updateScore(operation[1])
                assert game.getScore() == operation[2]
            elif operation[0] == 'set':
                game.setScore(operation[1])
                assert game.getScore() == operation[2]


class TestGamePace:
    """Test Game pace functionality."""
    
    def test_initial_pace(self):
        """Test initial pace is zero."""
        game = Game(headless=True)
        assert game.getPace() == 0
    
    def test_set_pace(self):
        """Test setting pace value."""
        game = Game(headless=True)
        
        game.setPace(3)
        assert game.getPace() == 3
        
        game.setPace(0)
        assert game.getPace() == 0
        
        game.setPace(10)
        assert game.getPace() == 10
    
    def test_pace_progression(self):
        """Test pace progression through game time."""
        game = Game(headless=True)
        
        # Simulate pace increases
        for expected_pace in range(1, 6):
            game.setPace(expected_pace)
            assert game.getPace() == expected_pace


class TestGameState:
    """Test Game state management."""
    
    def test_initial_state(self):
        """Test initial game state."""
        game = Game(headless=True)
        
        assert game.isActive()
        assert game.isPlaying()
        assert not game.isPaused()
    
    def test_pause_functionality(self):
        """Test pause/unpause functionality."""
        with full_pygame_mocks():
            game = Game(headless=True)
            # Create a mock player for pause functionality
            mock_player = Mock()
            mock_player.setCursor = Mock()
            
            # Initially not paused
            assert not game.isPaused()
            
            # Pause the game
            game.changePaused(mock_player)
            assert game.isPaused()
            
            # Unpause the game
            game.changePaused(mock_player)
            assert not game.isPaused()
    
    def test_end_game(self):
        """Test ending the game."""
        game = Game(headless=True)
        
        assert game.isActive()
        
        game.end()
        
        assert not game.isActive()
    
    def test_quit_game(self):
        """Test quitting the game."""
        game = Game(headless=True)
        
        assert game.isPlaying()
        
        game.quit()
        
        assert not game.isPlaying()
    
    def test_reset_game(self):
        """Test resetting the game."""
        game = Game(headless=True)
        
        # Modify game state
        game.setScore(50)
        game.setPace(5)
        game.end()
        
        # Reset
        game.reset()
        
        # Check state is reset
        assert game.getScore() == 0
        assert game.getPace() == 0
        assert game.isActive()


class TestGameClock:
    """Test Game clock integration."""
    
    def test_get_clock(self):
        """Test getting the game clock."""
        game = Game(headless=True)
        
        clock = game.getClock()
        
        assert clock is not None
        assert isinstance(clock, Clock)
    
    def test_clock_functionality(self):
        """Test clock integration with game."""
        game = Game(headless=True)
        clock = game.getClock()
        
        # Clock should be initialized
        assert clock.getMillis() >= 0
        assert clock.getTicks() >= 0


class TestGameDisplay:
    """Test Game display functionality."""
    
    def test_get_screen_headless(self):
        """Test getting screen in headless mode."""
        game = Game(headless=True)
        
        screen = game.getScreen()
        assert screen is None
    
    def test_get_screen_with_display(self):
        """Test getting screen with display mode."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            
            screen = game.getScreen()
            assert screen is not None
    
    def test_update_screen_headless(self):
        """Test screen update in headless mode."""
        game = Game(headless=True)
        mock_player = Mock()
        mock_lines = []
        
        # Should not raise exception in headless mode
        game.updateScreen(mock_player, mock_lines)
    
    def test_update_screen_with_display(self):
        """Test screen update with display mode."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            
            mock_player = Mock()
            mock_player.getCursor.return_value = mocks['image']['loaded_surface']
            mock_player.getPosition.return_value = (100, 100)
            
            mock_lines = []
            
            # Should call display functions
            game.updateScreen(mock_player, mock_lines)
            
            # Verify screen operations were called
            assert len(mocks['display']['surface'].fills) > 0
            assert len(mocks['display']['surface'].blits) > 0
    
    @pytest.mark.parametrize("visible", [True, False])
    def test_seed_is_only_rendered_when_visible(self, visible):
        """The HUD shows the seed only when it was passed explicitly."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            game.setSeed(4321)
            game.setSeedVisible(visible)

            mock_player = Mock()
            mock_player.getCursor.return_value = mocks['image']['loaded_surface']
            mock_player.getPosition.return_value = (100, 100)

            game.updateScreen(mock_player, [])

            texts = [t['text'] for t in mocks['font']['font_instance'].rendered_texts]
            assert ("Seed: 4321" in texts) is visible

    def test_print_end_display(self):
        """Test end game display."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            
            # Should not raise exception
            game.printEndDisplay()
            
            # Verify display operations
            assert len(mocks['display']['surface'].fills) > 0
            assert mocks['display']['flip'].called


class TestGameCleanup:
    """Test Game cleanup functionality."""
    
    def test_cleanup_headless(self):
        """Test cleanup in headless mode."""
        game = Game(headless=True)
        
        # Should not raise exception
        game.cleanup()
    
    def test_cleanup_with_display(self):
        """Test cleanup with display mode."""
        with patch('pygame.quit') as mock_quit:
            game = Game(headless=False)
            
            game.cleanup()
            
            # Should call pygame.quit
            mock_quit.assert_called_once()


class TestGamePauseIntegration:
    """Test Game pause integration with player."""
    
    def test_pause_changes_player_sprite(self):
        """Test that pausing changes player sprite."""
        with full_pygame_mocks():
            game = Game(headless=True)
            mock_player = Mock()
            
            # Pause the game
            game.changePaused(mock_player)
            
            # Should call setCursor on player
            mock_player.setCursor.assert_called()
            
            # Unpause
            game.changePaused(mock_player)
            
            # Should call setCursor again
            assert mock_player.setCursor.call_count == 2
    
    def test_pause_affects_color(self):
        """Test that pausing affects foreground color."""
        game = Game(headless=True)
        mock_player = Mock()
        
        original_color = game.FG_COLOR
        
        # Pause
        game.changePaused(mock_player)
        paused_color = game.FG_COLOR
        
        # Unpause
        game.changePaused(mock_player)
        unpaused_color = game.FG_COLOR
        
        # Colors should be different when paused
        assert paused_color != original_color
        assert unpaused_color == original_color


class TestGameBoundaryConstants:
    """Test Game boundary constants and validation."""
    
    def test_boundary_constants_valid(self):
        """Test that boundary constants are valid."""
        game = Game(headless=True)
        
        # Width and height should be positive
        assert game.WIDTH > 0
        assert game.HEIGHT > 0
        
        # Boundaries should make sense
        assert game.X_MIN < game.X_MAX
        assert game.Y_MIN < game.Y_MAX
        
        # Boundaries should be within screen dimensions
        assert game.X_MAX <= game.WIDTH
        assert game.Y_MAX <= game.HEIGHT
    
    def test_score_increment_valid(self):
        """Test that score increment is valid."""
        game = Game(headless=True)
        
        assert game.SCORE_INCREMENT > 0
        assert isinstance(game.SCORE_INCREMENT, int)


class TestGameEdgeCases:
    """Test Game edge cases and error conditions."""
    
    def test_multiple_initializations(self):
        """Test multiple game initializations."""
        # Should be able to create multiple game instances
        game1 = Game(headless=True)
        game2 = Game(headless=True)
        
        assert game1.getScore() == 0
        assert game2.getScore() == 0
        
        # Modifying one shouldn't affect the other
        game1.setScore(10)
        assert game2.getScore() == 0
    
    def test_extreme_score_values(self):
        """Test game with extreme score values."""
        game = Game(headless=True)
        
        # Test very large positive score
        large_score = 999999
        game.setScore(large_score)
        assert game.getScore() == large_score
        
        # Test very large negative score
        negative_score = -999999
        game.setScore(negative_score)
        assert game.getScore() == negative_score
    
    def test_rapid_state_changes(self):
        """Test rapid state changes."""
        game = Game(headless=True)
        mock_player = Mock()
        
        # Rapid pause/unpause
        for _ in range(100):
            game.changePaused(mock_player)
        
        # Should end up unpaused (started unpaused, 100 toggles = even)
        assert not game.isPaused()
    
    def test_reset_after_end(self):
        """Test reset after game has ended."""
        game = Game(headless=True)
        
        # Modify state and end game
        game.setScore(50)
        game.setPace(3)
        game.end()
        
        assert not game.isActive()
        
        # Reset should restore active state
        game.reset()
        
        assert game.isActive()
        assert game.getScore() == 0
        assert game.getPace() == 0


@pytest.mark.integration
class TestGameIntegration:
    """Integration tests for Game with other components."""
    
    def test_game_with_real_clock(self):
        """Test game integration with real clock."""
        game = Game(headless=True)
        clock = game.getClock()
        
        # Clock should be functional
        initial_millis = clock.getMillis()
        clock.update()
        updated_millis = clock.getMillis()
        
        # Time should have progressed
        assert updated_millis >= initial_millis
    
    def test_game_state_persistence(self):
        """Test that game state persists correctly."""
        game = Game(headless=True)
        
        # Set up some state
        game.setScore(25)
        game.setPace(2)
        
        # State should persist
        assert game.getScore() == 25
        assert game.getPace() == 2
        
        # After operations, state should be maintained
        game.incrementScore()
        assert game.getScore() == 26
        assert game.getPace() == 2  # Should not change


@pytest.mark.performance
class TestGamePerformance:
    """Performance tests for Game operations."""
    
    def test_score_operations_performance(self):
        """Test performance of score operations."""
        game = Game(headless=True)
        
        import time
        start_time = time.time()
        
        # Perform many score operations
        for i in range(10000):
            if i % 3 == 0:
                game.incrementScore()
            elif i % 3 == 1:
                game.decrementScore()
            else:
                game.updateScore(1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 1.0, f"Score operations took too long: {duration}s"
    
    def test_state_change_performance(self):
        """Test performance of state changes."""
        game = Game(headless=True)
        mock_player = Mock()
        
        import time
        start_time = time.time()
        
        # Perform many state changes
        for _ in range(1000):
            game.changePaused(mock_player)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 1.0, f"State changes took too long: {duration}s"
